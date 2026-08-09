"""Installed extension discovery, provenance checks, and activation.

Configuration parsing is deliberately separate from this module.  Preparing a
plan reads distribution metadata but never imports extension code; activation
is the only operation that imports the selected aggregate modules.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from importlib import import_module, metadata
from threading import RLock
from typing import cast

from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version

from stochaflow.utils.config import StochaflowConfig

EXTENSION_ENTRY_POINT_GROUP = "stochaflow.extensions"

_ENTRY_POINT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ACCEPTANCE_METHODS = frozenset({"library-policy", "prompt", "force-flag"})


class ExtensionPluginError(RuntimeError):
    """Base error for extension discovery, validation, and activation."""


class ExtensionDiscoveryError(ExtensionPluginError):
    """Raised when installed entry-point metadata cannot form a selection."""


class ExtensionIdentityError(ExtensionPluginError):
    """Raised when expected and installed plugin identities are incompatible."""


class ExtensionVersionMismatchError(ExtensionPluginError):
    """Raised when plugin versions differ and the active policy rejects them."""

    def __init__(self, mismatches: Sequence[ExtensionVersionMismatch]) -> None:
        self.mismatches = tuple(mismatches)
        details = "; ".join(
            f"{item.current.name}: expected {item.expected.version}, "
            f"installed {item.current.version}"
            for item in self.mismatches
        )
        super().__init__(
            "extension plugin version mismatch"
            + (f": {details}" if details else "")
        )


class ExtensionActivationError(ExtensionPluginError):
    """Raised when selected extension code cannot be activated."""


class ExtensionActivationStateError(ExtensionActivationError):
    """Raised when the process-wide activation lifecycle cannot proceed."""


class ExtensionVersionPolicy(StrEnum):
    """Programmatic policy for preflighted plugin version differences."""

    REJECT = "reject"
    ALLOW = "allow"


class ExtensionSelectionPolicy(StrEnum):
    """How a current plugin selection is compared with expected provenance."""

    EXACT = "exact"
    INTERSECTION = "intersection"


@dataclass(frozen=True, slots=True)
class ExtensionPluginProvenance:
    """Installed distribution identity for one selected extension entry point."""

    name: str
    distribution: str
    version: str
    target: str

    def __post_init__(self) -> None:
        _validate_entry_point_name(self.name, context="plugin provenance name")
        canonical_distribution = _canonical_distribution_name(
            self.distribution,
            context=f"plugin '{self.name}' distribution",
        )
        if self.distribution != canonical_distribution:
            raise ExtensionIdentityError(
                f"plugin '{self.name}' distribution must be canonical "
                f"'{canonical_distribution}', got '{self.distribution}'"
            )
        _parse_version(
            self.version,
            context=f"plugin '{self.name}' distribution version",
        )
        _validate_module_target(
            self.target,
            context=f"plugin '{self.name}' entry-point target",
        )


@dataclass(frozen=True, slots=True)
class ExtensionVersionMismatch:
    """Expected and installed provenance whose identities match but versions do not."""

    expected: ExtensionPluginProvenance
    current: ExtensionPluginProvenance


@dataclass(frozen=True, slots=True)
class ExtensionVersionAcceptance:
    """Audit record for one explicitly accepted version mismatch."""

    expected: ExtensionPluginProvenance
    current: ExtensionPluginProvenance
    method: str


@dataclass(frozen=True, slots=True, init=False)
class ExtensionActivationPlan:
    """Side-effect-free metadata preflight with a sealed config snapshot."""

    _config_snapshot: StochaflowConfig = field(repr=False)
    provenance: tuple[ExtensionPluginProvenance, ...]
    version_mismatches: tuple[ExtensionVersionMismatch, ...]
    selection_policy: ExtensionSelectionPolicy
    _activation_receipt_token: object = field(
        default_factory=object,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        config: StochaflowConfig,
        provenance: tuple[ExtensionPluginProvenance, ...],
        version_mismatches: tuple[ExtensionVersionMismatch, ...],
        selection_policy: ExtensionSelectionPolicy,
    ) -> None:
        config_snapshot = deepcopy(config)
        config_snapshot.validate()
        object.__setattr__(self, "_config_snapshot", config_snapshot)
        object.__setattr__(self, "provenance", tuple(provenance))
        object.__setattr__(self, "version_mismatches", tuple(version_mismatches))
        object.__setattr__(self, "selection_policy", selection_policy)
        object.__setattr__(self, "_activation_receipt_token", object())

    @property
    def config(self) -> StochaflowConfig:
        """Return a detached copy of the sealed preflight snapshot."""

        return deepcopy(self._config_snapshot)


@dataclass(frozen=True, slots=True)
class ResolvedExtensions:
    """Materialized configuration and provenance produced by activation."""

    config: StochaflowConfig
    provenance: tuple[ExtensionPluginProvenance, ...]
    acceptance_audit: tuple[ExtensionVersionAcceptance, ...]
    _activation_receipt_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", deepcopy(self.config))


class PluginActivationState(Enum):
    UNACTIVATED = "unactivated"
    ACTIVATING = "activating"
    ACTIVE = "active"
    FAILED = "failed"


@dataclass(slots=True)
class PluginActivationRuntime:
    state: PluginActivationState = PluginActivationState.UNACTIVATED
    selection: tuple[ExtensionPluginProvenance, ...] | None = None
    failure: BaseException | None = None


_activation_lock = RLock()
_activation_runtime = PluginActivationRuntime()


def _validate_entry_point_name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not _ENTRY_POINT_NAME_PATTERN.fullmatch(value):
        raise ExtensionIdentityError(
            f"{context} must be a valid, non-empty entry-point name"
        )
    return value


def _canonical_distribution_name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExtensionIdentityError(f"{context} must be a non-empty string")
    try:
        return canonicalize_name(value, validate=True)
    except (InvalidName, TypeError) as exc:
        raise ExtensionIdentityError(
            f"{context} is not a valid distribution name: {value!r}"
        ) from exc


def _parse_version(value: object, *, context: str) -> Version:
    if not isinstance(value, str) or not value:
        raise ExtensionIdentityError(f"{context} must be a non-empty string")
    try:
        return Version(value)
    except InvalidVersion as exc:
        raise ExtensionIdentityError(
            f"{context} is not a valid PEP 440 version: {value!r}"
        ) from exc


def _validate_module_target(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not part.isidentifier() for part in value.split("."))
    ):
        raise ExtensionIdentityError(
            f"{context} must be a pure Python module without a callable or extras"
        )
    return value


def _provenance_sort_key(
    item: ExtensionPluginProvenance,
) -> tuple[str, str, str]:
    return item.name, item.distribution, item.target


def _validate_provenance_sequence(
    values: Sequence[ExtensionPluginProvenance],
    *,
    context: str,
) -> tuple[ExtensionPluginProvenance, ...]:
    result: list[ExtensionPluginProvenance] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        value_object = cast(object, value)
        if not isinstance(value_object, ExtensionPluginProvenance):
            raise ExtensionIdentityError(
                f"{context}[{index}] must be ExtensionPluginProvenance"
            )
        # Reconstructing also validates objects produced by unsafe deserialization.
        validated = ExtensionPluginProvenance(
            name=value.name,
            distribution=value.distribution,
            version=value.version,
            target=value.target,
        )
        if validated.name in seen:
            raise ExtensionIdentityError(
                f"{context} contains duplicate plugin name '{validated.name}'"
            )
        seen.add(validated.name)
        result.append(validated)
    return tuple(sorted(result, key=_provenance_sort_key))


def parse_extension_plugin_provenance(
    raw: object,
) -> tuple[ExtensionPluginProvenance, ...]:
    """Strictly parse checkpoint extension provenance into immutable records."""

    if not isinstance(raw, (list, tuple)):
        raise ExtensionIdentityError("extension_plugins must be a list")
    required_fields = {"name", "distribution", "version", "target"}
    parsed: list[ExtensionPluginProvenance] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ExtensionIdentityError(
                f"extension_plugins[{index}] must be a mapping"
            )
        fields = set(item)
        if fields != required_fields:
            missing = sorted(required_fields - fields)
            unknown = sorted(fields - required_fields, key=str)
            details: list[str] = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unknown:
                details.append(
                    "unknown " + ", ".join(repr(field) for field in unknown)
                )
            raise ExtensionIdentityError(
                f"extension_plugins[{index}] has invalid fields: "
                + "; ".join(details)
            )
        parsed.append(
            ExtensionPluginProvenance(
                name=item["name"],
                distribution=item["distribution"],
                version=item["version"],
                target=item["target"],
            )
        )
    return _validate_provenance_sequence(parsed, context="extension_plugins")


def extension_plugin_provenance_to_dicts(
    provenance: Sequence[ExtensionPluginProvenance],
) -> list[dict[str, str]]:
    """Serialize validated provenance into checkpoint-safe dictionaries."""

    validated = _validate_provenance_sequence(
        provenance,
        context="extension_plugins",
    )
    return [
        {
            "name": item.name,
            "distribution": item.distribution,
            "version": item.version,
            "target": item.target,
        }
        for item in validated
    ]


def _entry_point_provenance(entry_point: metadata.EntryPoint) -> ExtensionPluginProvenance:
    declared_name: object = "<unavailable>"
    declared_target: object = "<unavailable>"
    declared_distribution: object = "<unavailable>"
    declared_version: object = "<unavailable>"
    try:
        declared_name = entry_point.name
        declared_target = entry_point.value
        name = _validate_entry_point_name(
            declared_name,
            context="extension entry-point name",
        )
        target = _validate_module_target(
            declared_target,
            context=f"extension entry point '{name}' target",
        )
        distribution = entry_point.dist
        if distribution is None:
            raise ExtensionDiscoveryError(
                f"extension entry point '{name}' has no owning distribution"
            )
        declared_distribution = distribution.metadata.get("Name")
        distribution_name = _canonical_distribution_name(
            declared_distribution,
            context=f"extension entry point '{name}' distribution",
        )
        declared_version = distribution.version
        _parse_version(
            declared_version,
            context=f"extension entry point '{name}' distribution version",
        )
        return ExtensionPluginProvenance(
            name=name,
            distribution=distribution_name,
            version=declared_version,
            target=target,
        )
    except Exception as exc:
        raise ExtensionDiscoveryError(
            "invalid extension entry-point metadata "
            f"(name={declared_name!r}, distribution={declared_distribution!r}, "
            f"version={declared_version!r}, target={declared_target!r}): {exc}"
        ) from exc


def _discover_selected_plugins(
    selected_names: tuple[str, ...] | None,
    *,
    expected_by_name: Mapping[str, ExtensionPluginProvenance] | None = None,
) -> tuple[ExtensionPluginProvenance, ...]:
    if selected_names == ():
        return ()
    try:
        installed = tuple(metadata.entry_points(group=EXTENSION_ENTRY_POINT_GROUP))
    except Exception as exc:
        raise ExtensionDiscoveryError(
            f"failed to discover '{EXTENSION_ENTRY_POINT_GROUP}' entry points: {exc}"
        ) from exc

    selected_set = set(selected_names) if selected_names is not None else None
    candidates: list[metadata.EntryPoint] = []
    for entry_point in installed:
        try:
            name = entry_point.name
        except Exception as exc:
            if selected_set is None:
                raise ExtensionDiscoveryError(
                    f"failed to read an extension entry-point name: {exc}"
                ) from exc
            continue
        if selected_set is None:
            _validate_entry_point_name(name, context="extension entry-point name")
        if selected_set is None or name in selected_set:
            candidates.append(entry_point)

    by_name: dict[str, list[metadata.EntryPoint]] = {}
    for entry_point in candidates:
        by_name.setdefault(entry_point.name, []).append(entry_point)
    duplicates = sorted(name for name, entries in by_name.items() if len(entries) > 1)
    if duplicates:
        raise ExtensionDiscoveryError(
            "multiple installed distributions declare selected extension "
            f"entry-point name(s): {', '.join(duplicates)}"
        )

    if selected_names is not None:
        missing = sorted(set(selected_names) - set(by_name))
        if missing:
            raise ExtensionDiscoveryError(
                _missing_plugin_message(
                    missing,
                    expected_by_name=expected_by_name or {},
                )
            )

    provenance = tuple(_entry_point_provenance(entry) for entry in candidates)
    return tuple(sorted(provenance, key=_provenance_sort_key))


def _missing_plugin_message(
    missing: Sequence[str],
    *,
    expected_by_name: Mapping[str, ExtensionPluginProvenance],
) -> str:
    """Return package-manager-neutral diagnostics for missing entry points."""

    details: list[str] = ["requested extension plugin(s) are not installed:"]
    suggestions: list[str] = []
    for name in missing:
        expected = expected_by_name.get(name)
        if expected is None:
            details.append(
                f"- entry-point name={name!r}, distribution=<unavailable>, "
                "target=<unavailable> (no checkpoint provenance)"
            )
            suggestions.append(
                f"Install the distribution that declares entry point {name!r} "
                f"in group {EXTENSION_ENTRY_POINT_GROUP!r} into this Python "
                "environment, then retry."
            )
        else:
            details.append(
                f"- entry-point name={name!r}, expected "
                f"distribution={expected.distribution!r}, "
                f"version={expected.version!r}, target={expected.target!r}"
            )
            suggestions.append(
                f"Install distribution {expected.distribution!r} into the Python "
                "environment used by this Stochaflow CLI so it provides entry "
                f"point {name!r}, then retry."
            )
    details.append(f"Current Python executable: {sys.executable}")
    details.extend(suggestions)
    return "\n".join(details)


def _compare_expected_provenance(
    current: tuple[ExtensionPluginProvenance, ...],
    expected: tuple[ExtensionPluginProvenance, ...],
    *,
    selection_policy: ExtensionSelectionPolicy,
) -> tuple[ExtensionVersionMismatch, ...]:
    current_by_name = {item.name: item for item in current}
    expected_by_name = {item.name: item for item in expected}
    if selection_policy is ExtensionSelectionPolicy.EXACT:
        missing = sorted(set(expected_by_name) - set(current_by_name))
        unexpected = sorted(set(current_by_name) - set(expected_by_name))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing expected plugin(s): {', '.join(missing)}")
            if unexpected:
                details.append(
                    f"unexpected selected plugin(s): {', '.join(unexpected)}"
                )
            raise ExtensionIdentityError(
                "extension plugin selection does not match checkpoint provenance: "
                + "; ".join(details)
            )
        names = sorted(current_by_name)
    else:
        names = sorted(set(current_by_name) & set(expected_by_name))

    mismatches: list[ExtensionVersionMismatch] = []
    for name in names:
        current_item = current_by_name[name]
        expected_item = expected_by_name[name]
        if (
            current_item.distribution != expected_item.distribution
            or current_item.target != expected_item.target
        ):
            raise ExtensionIdentityError(
                f"extension plugin '{name}' identity differs from checkpoint: "
                f"expected distribution={expected_item.distribution!r}, "
                f"target={expected_item.target!r}; installed "
                f"distribution={current_item.distribution!r}, "
                f"target={current_item.target!r}"
            )
        if _parse_version(
            current_item.version,
            context=f"plugin '{name}' installed version",
        ) != _parse_version(
            expected_item.version,
            context=f"plugin '{name}' expected version",
        ):
            mismatches.append(
                ExtensionVersionMismatch(
                    expected=expected_item,
                    current=current_item,
                )
            )
    return tuple(mismatches)


def prepare_extension_plugins(
    config: StochaflowConfig,
    *,
    expected_provenance: Sequence[ExtensionPluginProvenance] | None = None,
    selection_policy: ExtensionSelectionPolicy = ExtensionSelectionPolicy.EXACT,
) -> ExtensionActivationPlan:
    """Discover and preflight selected plugins without importing their code."""

    config_copy = deepcopy(config)
    config_copy.validate()
    declared = config_copy.extensions.plugins
    selected_names = None if declared is None else tuple(declared)
    expected: tuple[ExtensionPluginProvenance, ...] | None = None
    if expected_provenance is not None:
        expected = _validate_provenance_sequence(
            expected_provenance,
            context="expected_provenance",
        )
    current = _discover_selected_plugins(
        selected_names,
        expected_by_name=(
            {item.name: item for item in expected} if expected is not None else None
        ),
    )
    if (
        selected_names is None
        and expected is not None
        and selection_policy is ExtensionSelectionPolicy.EXACT
    ):
        expected_by_name = {item.name: item for item in expected}
        current_names = {item.name for item in current}
        missing = sorted(set(expected_by_name) - current_names)
        if missing:
            raise ExtensionDiscoveryError(
                _missing_plugin_message(
                    missing,
                    expected_by_name=expected_by_name,
                )
            )

    mismatches: tuple[ExtensionVersionMismatch, ...] = ()
    if expected is not None:
        mismatches = _compare_expected_provenance(
            current,
            expected,
            selection_policy=selection_policy,
        )
    return ExtensionActivationPlan(
        config=config_copy,
        provenance=current,
        version_mismatches=mismatches,
        selection_policy=selection_policy,
    )


def _accepted_mismatches(
    plan: ExtensionActivationPlan,
    *,
    policy: ExtensionVersionPolicy,
    acceptance_method: str | None,
) -> tuple[ExtensionVersionAcceptance, ...]:
    if policy is ExtensionVersionPolicy.REJECT:
        if acceptance_method is not None:
            raise ValueError("acceptance_method is only valid with ALLOW policy")
        if plan.version_mismatches:
            raise ExtensionVersionMismatchError(plan.version_mismatches)
        return ()

    method = "library-policy" if acceptance_method is None else acceptance_method
    if method not in _ACCEPTANCE_METHODS:
        choices = ", ".join(sorted(_ACCEPTANCE_METHODS))
        raise ValueError(f"acceptance_method must be one of: {choices}")
    return tuple(
        ExtensionVersionAcceptance(
            expected=mismatch.expected,
            current=mismatch.current,
            method=method,
        )
        for mismatch in plan.version_mismatches
    )


def _materialized_config(plan: ExtensionActivationPlan) -> StochaflowConfig:
    config = plan.config
    config.extensions.plugins = [item.name for item in plan.provenance]
    return config


def activate_extension_plugins(
    plan: ExtensionActivationPlan,
    *,
    policy: ExtensionVersionPolicy = ExtensionVersionPolicy.REJECT,
    acceptance_method: str | None = None,
) -> ResolvedExtensions:
    """Import a preflighted selection once and return its resolved config.

    Activation is process-wide and intentionally irreversible because decorator
    registrations cannot be rolled back safely.  A failed or re-entrant import
    poisons the activation state and requires a fresh Python process.
    """

    acceptance_audit = _accepted_mismatches(
        plan,
        policy=policy,
        acceptance_method=acceptance_method,
    )
    selection = tuple(plan.provenance)

    with _activation_lock:
        if _activation_runtime.state is PluginActivationState.FAILED:
            raise ExtensionActivationStateError(
                "extension activation previously failed; restart the Python process"
            ) from _activation_runtime.failure
        if _activation_runtime.state is PluginActivationState.ACTIVATING:
            error = ExtensionActivationStateError(
                "re-entrant extension activation is not supported; restart the "
                "Python process"
            )
            _activation_runtime.state = PluginActivationState.FAILED
            _activation_runtime.failure = error
            raise error
        if _activation_runtime.state is PluginActivationState.ACTIVE:
            if selection != _activation_runtime.selection:
                raise ExtensionActivationStateError(
                    "a different extension plugin selection is already active in "
                    "this Python process"
                )
        else:
            _activation_runtime.state = PluginActivationState.ACTIVATING
            active_plugin: ExtensionPluginProvenance | None = None
            try:
                for plugin in selection:
                    active_plugin = plugin
                    import_module(plugin.target)
            except BaseException as exc:
                if isinstance(exc, ExtensionPluginError):
                    error = exc
                else:
                    error = ExtensionActivationError(
                        "failed to activate extension plugin "
                        f"'{active_plugin.name if active_plugin else '<unknown>'}' "
                        "(distribution="
                        f"{active_plugin.distribution if active_plugin else '<unknown>'!r}, "
                        "version="
                        f"{active_plugin.version if active_plugin else '<unknown>'!r}, "
                        "target="
                        f"{active_plugin.target if active_plugin else '<unknown>'!r}): "
                        f"{exc}"
                    )
                _activation_runtime.state = PluginActivationState.FAILED
                _activation_runtime.failure = error
                if error is exc:
                    raise
                raise error from exc
            _activation_runtime.selection = selection
            _activation_runtime.state = PluginActivationState.ACTIVE

    return ResolvedExtensions(
        config=_materialized_config(plan),
        provenance=selection,
        acceptance_audit=acceptance_audit,
        _activation_receipt_token=plan._activation_receipt_token,
    )


def require_resolved_extensions_for_plan(
    plan: ExtensionActivationPlan,
    extensions: ResolvedExtensions,
) -> None:
    """Reject an activation receipt produced for a different preflight plan."""

    if not isinstance(cast(object, plan), ExtensionActivationPlan):
        raise TypeError("extension activation plan must be ExtensionActivationPlan")
    if not isinstance(cast(object, extensions), ResolvedExtensions):
        raise TypeError("resolved extensions must be ResolvedExtensions")
    if extensions._activation_receipt_token is not plan._activation_receipt_token:
        raise ValueError(
            "resolved extensions activation receipt belongs to a different "
            "extension plan"
        )
    if extensions.provenance != plan.provenance:
        raise ValueError(
            "resolved extensions provenance does not match its activation plan"
        )
    if extensions.config != _materialized_config(plan):
        raise ValueError(
            "resolved extensions config does not match its activation plan"
        )


__all__ = [
    "EXTENSION_ENTRY_POINT_GROUP",
    "ExtensionActivationError",
    "ExtensionActivationPlan",
    "ExtensionActivationStateError",
    "ExtensionDiscoveryError",
    "ExtensionIdentityError",
    "ExtensionPluginError",
    "ExtensionPluginProvenance",
    "ExtensionSelectionPolicy",
    "ExtensionVersionAcceptance",
    "ExtensionVersionMismatch",
    "ExtensionVersionMismatchError",
    "ExtensionVersionPolicy",
    "ResolvedExtensions",
    "activate_extension_plugins",
    "extension_plugin_provenance_to_dicts",
    "parse_extension_plugin_provenance",
    "prepare_extension_plugins",
    "require_resolved_extensions_for_plan",
]

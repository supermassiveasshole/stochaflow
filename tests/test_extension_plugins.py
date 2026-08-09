"""Tests for installed extension discovery, preflight, and activation."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import stochaflow.utils.plugins as plugin_runtime
from stochaflow.utils.config import load_config_dict
from stochaflow.utils.plugins import (
    ExtensionActivationError,
    ExtensionActivationStateError,
    ExtensionDiscoveryError,
    ExtensionIdentityError,
    ExtensionPluginProvenance,
    ExtensionSelectionPolicy,
    ExtensionVersionMismatchError,
    ExtensionVersionPolicy,
    activate_extension_plugins,
    extension_plugin_provenance_to_dicts,
    parse_extension_plugin_provenance,
    prepare_extension_plugins,
)


@dataclass
class FakeDistribution:
    name: str
    version: str

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    distribution: str = "example-project"
    version: str = "1.0"

    @property
    def dist(self) -> FakeDistribution:
        return FakeDistribution(self.distribution, self.version)


pytestmark = pytest.mark.usefixtures("isolated_extension_activation_state")


def _config(*, plugins: list[str] | None) -> Any:
    return load_config_dict(
        {
            "experiment": {"name": "plugin-test"},
            "extensions": {"plugins": plugins},
            "data": {"name": "test-data", "params": {}},
            "model": {"name": "test-model", "params": {}},
            "training": {"name": "test-training", "params": {}},
        }
    )


def _install_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    *entry_points: FakeEntryPoint,
) -> None:
    def discover(*, group: str) -> tuple[FakeEntryPoint, ...]:
        assert group == "stochaflow.extensions"
        return tuple(entry_points)

    monkeypatch.setattr(plugin_runtime.metadata, "entry_points", discover)


def _provenance(
    *,
    name: str = "example",
    distribution: str = "example-project",
    version: str = "1.0",
    target: str = "example_project.extension",
) -> ExtensionPluginProvenance:
    return ExtensionPluginProvenance(
        name=name,
        distribution=distribution,
        version=version,
        target=target,
    )


def test_empty_selection_does_not_discover_or_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        plugin_runtime.metadata,
        "entry_points",
        lambda **_: pytest.fail("empty selection must not inspect the environment"),
    )
    monkeypatch.setattr(
        plugin_runtime,
        "import_module",
        lambda _: pytest.fail("empty selection must not import code"),
    )

    plan = prepare_extension_plugins(_config(plugins=[]))
    resolved = activate_extension_plugins(plan)

    assert plan.provenance == ()
    assert resolved.config.extensions.plugins == []
    assert resolved.provenance == ()
    assert resolved.acceptance_audit == ()


def test_null_discovers_all_and_materializes_deterministic_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("zeta", "zeta.extension", "Zeta_Project", "2.0"),
        FakeEntryPoint("alpha", "alpha.extension", "Alpha.Project", "1.0"),
    )
    imported: list[str] = []
    monkeypatch.setattr(plugin_runtime, "import_module", imported.append)
    config = _config(plugins=None)

    plan = prepare_extension_plugins(config)
    resolved = activate_extension_plugins(plan)

    assert [item.name for item in plan.provenance] == ["alpha", "zeta"]
    assert [item.distribution for item in plan.provenance] == [
        "alpha-project",
        "zeta-project",
    ]
    assert resolved.config.extensions.plugins == ["alpha", "zeta"]
    assert config.extensions.plugins is None
    assert imported == ["alpha.extension", "zeta.extension"]


def test_explicit_selection_ignores_unselected_broken_and_duplicate_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("selected", "selected.extension"),
        FakeEntryPoint("broken", "not-a-module:factory"),
        FakeEntryPoint("duplicate", "one.extension"),
        FakeEntryPoint("duplicate", "two.extension"),
    )

    plan = prepare_extension_plugins(_config(plugins=["selected"]))

    assert [item.name for item in plan.provenance] == ["selected"]


def test_all_selection_validates_every_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("selected", "selected.extension"),
        FakeEntryPoint("broken", "not-a-module:factory"),
    )

    with pytest.raises(ExtensionDiscoveryError, match="pure Python module"):
        prepare_extension_plugins(_config(plugins=None))


def test_selected_duplicate_and_missing_names_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("duplicate", "one.extension", "one"),
        FakeEntryPoint("duplicate", "two.extension", "two"),
    )
    with pytest.raises(ExtensionDiscoveryError, match="multiple installed"):
        prepare_extension_plugins(_config(plugins=["duplicate"]))

    _install_entry_points(monkeypatch)
    with pytest.raises(ExtensionDiscoveryError, match="not installed"):
        prepare_extension_plugins(_config(plugins=["missing"]))


def test_fresh_missing_plugin_reports_unknown_identity_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch)

    with pytest.raises(ExtensionDiscoveryError) as captured:
        prepare_extension_plugins(_config(plugins=["missing"]))

    message = str(captured.value)
    assert "entry-point name='missing'" in message
    assert "distribution=<unavailable>" in message
    assert "target=<unavailable> (no checkpoint provenance)" in message
    assert f"Current Python executable: {sys.executable}" in message
    assert "distribution that declares entry point 'missing'" in message
    assert "pip install" not in message
    assert "uv add" not in message


def test_checkpoint_missing_plugin_reports_expected_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch)
    expected = _provenance(
        name="physics",
        distribution="physics-extension",
        version="2.4",
        target="physics_extension.stochaflow_ext",
    )

    with pytest.raises(ExtensionDiscoveryError) as captured:
        prepare_extension_plugins(
            _config(plugins=["physics"]),
            expected_provenance=[expected],
        )

    message = str(captured.value)
    assert "entry-point name='physics'" in message
    assert "expected distribution='physics-extension'" in message
    assert "version='2.4'" in message
    assert "target='physics_extension.stochaflow_ext'" in message
    assert f"Current Python executable: {sys.executable}" in message
    assert "Install distribution 'physics-extension'" in message


def test_discover_all_exact_reports_missing_checkpoint_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch)
    expected = _provenance(
        name="physics",
        distribution="physics-extension",
        version="2.4",
        target="physics_extension.stochaflow_ext",
    )

    with pytest.raises(ExtensionDiscoveryError) as captured:
        prepare_extension_plugins(
            _config(plugins=None),
            expected_provenance=[expected],
            selection_policy=ExtensionSelectionPolicy.EXACT,
        )

    message = str(captured.value)
    assert "entry-point name='physics'" in message
    assert "expected distribution='physics-extension'" in message
    assert f"Current Python executable: {sys.executable}" in message


def test_discover_all_intersection_allows_absent_checkpoint_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch)
    expected = _provenance(
        name="physics",
        distribution="physics-extension",
        target="physics_extension.stochaflow_ext",
    )

    plan = prepare_extension_plugins(
        _config(plugins=None),
        expected_provenance=[expected],
        selection_policy=ExtensionSelectionPolicy.INTERSECTION,
    )

    assert plan.provenance == ()
    assert plan.version_mismatches == ()


def test_intersection_missing_plugins_use_only_matching_checkpoint_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch)
    expected = [
        _provenance(
            name="shared",
            distribution="shared-extension",
            target="shared_extension.stochaflow_ext",
        ),
        _provenance(
            name="removed",
            distribution="removed-extension",
            target="removed_extension.stochaflow_ext",
        ),
    ]

    with pytest.raises(ExtensionDiscoveryError) as captured:
        prepare_extension_plugins(
            _config(plugins=["shared", "new"]),
            expected_provenance=expected,
            selection_policy=ExtensionSelectionPolicy.INTERSECTION,
        )

    message = str(captured.value)
    assert "entry-point name='shared'" in message
    assert "expected distribution='shared-extension'" in message
    assert "entry-point name='new'" in message
    assert "distribution=<unavailable>" in message
    assert "removed-extension" not in message


@pytest.mark.parametrize(
    "target",
    ["module:factory", "module [extra]", "module-name", " module"],
)
def test_selected_target_must_be_a_pure_module(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    _install_entry_points(monkeypatch, FakeEntryPoint("example", target))

    with pytest.raises(ExtensionDiscoveryError, match="pure Python module"):
        prepare_extension_plugins(_config(plugins=["example"]))


def test_prepare_validates_exact_identity_and_pep440_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("example", "example_project.extension", version="1.0.0"),
    )
    equivalent = _provenance(version="1.0")

    plan = prepare_extension_plugins(
        _config(plugins=["example"]),
        expected_provenance=[equivalent],
    )

    assert plan.version_mismatches == ()

    wrong_target = _provenance(target="old.extension")
    with pytest.raises(ExtensionIdentityError, match="identity differs"):
        prepare_extension_plugins(
            _config(plugins=["example"]),
            expected_provenance=[wrong_target],
        )


def test_exact_and_intersection_selection_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("current", "current.extension", "current", "1"),
        FakeEntryPoint("shared", "shared.extension", "shared", "2"),
    )
    expected = [
        _provenance(
            name="removed",
            distribution="removed",
            target="removed.extension",
        ),
        _provenance(
            name="shared",
            distribution="shared",
            version="1",
            target="shared.extension",
        ),
    ]
    config = _config(plugins=["current", "shared"])

    with pytest.raises(ExtensionIdentityError, match="selection does not match"):
        prepare_extension_plugins(config, expected_provenance=expected)

    plan = prepare_extension_plugins(
        config,
        expected_provenance=expected,
        selection_policy=ExtensionSelectionPolicy.INTERSECTION,
    )

    assert len(plan.version_mismatches) == 1
    assert plan.version_mismatches[0].current.name == "shared"


def test_version_policy_rejects_or_records_controlled_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("example", "example_project.extension", version="2.0"),
    )
    monkeypatch.setattr(plugin_runtime, "import_module", lambda _: None)
    plan = prepare_extension_plugins(
        _config(plugins=["example"]),
        expected_provenance=[_provenance(version="1.0")],
    )

    with pytest.raises(ExtensionVersionMismatchError) as error:
        activate_extension_plugins(plan)
    assert error.value.mismatches == plan.version_mismatches

    resolved = activate_extension_plugins(
        plan,
        policy=ExtensionVersionPolicy.ALLOW,
        acceptance_method="force-flag",
    )
    assert len(resolved.acceptance_audit) == 1
    assert resolved.acceptance_audit[0].method == "force-flag"


def test_allow_policy_defaults_to_library_audit_and_rejects_unknown_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("example", "example_project.extension", version="2.0"),
    )
    monkeypatch.setattr(plugin_runtime, "import_module", lambda _: None)
    plan = prepare_extension_plugins(
        _config(plugins=["example"]),
        expected_provenance=[_provenance(version="1.0")],
    )

    resolved = activate_extension_plugins(
        plan,
        policy=ExtensionVersionPolicy.ALLOW,
    )
    assert resolved.acceptance_audit[0].method == "library-policy"

    for invalid_method in ("", "anything"):
        with pytest.raises(ValueError, match="acceptance_method"):
            activate_extension_plugins(
                plan,
                policy=ExtensionVersionPolicy.ALLOW,
                acceptance_method=invalid_method,
            )


def test_same_selection_activation_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("example", "example_project.extension"),
    )
    imports: list[str] = []
    monkeypatch.setattr(plugin_runtime, "import_module", imports.append)
    plan = prepare_extension_plugins(_config(plugins=["example"]))

    first = activate_extension_plugins(plan)
    second = activate_extension_plugins(plan)

    assert imports == ["example_project.extension"]
    assert first.provenance == second.provenance


def test_active_process_rejects_different_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("first", "first.extension", "first"),
        FakeEntryPoint("second", "second.extension", "second"),
    )
    monkeypatch.setattr(plugin_runtime, "import_module", lambda _: None)
    activate_extension_plugins(
        prepare_extension_plugins(_config(plugins=["first"]))
    )

    with pytest.raises(ExtensionActivationStateError, match="different"):
        activate_extension_plugins(
            prepare_extension_plugins(_config(plugins=["second"]))
        )


def test_import_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("example", "example_project.extension"),
    )

    def fail(_: str) -> None:
        raise RuntimeError("broken import")

    monkeypatch.setattr(plugin_runtime, "import_module", fail)
    plan = prepare_extension_plugins(_config(plugins=["example"]))

    with pytest.raises(ExtensionActivationError, match="broken import"):
        activate_extension_plugins(plan)
    with pytest.raises(ExtensionActivationStateError, match="restart"):
        activate_extension_plugins(plan)


def test_reentrant_activation_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("example", "example_project.extension"),
    )
    plan = prepare_extension_plugins(_config(plugins=["example"]))

    def reenter(_: str) -> None:
        activate_extension_plugins(plan)

    monkeypatch.setattr(plugin_runtime, "import_module", reenter)

    with pytest.raises(ExtensionActivationStateError, match="re-entrant"):
        activate_extension_plugins(plan)
    with pytest.raises(ExtensionActivationStateError, match="restart"):
        activate_extension_plugins(plan)


def test_concurrent_same_selection_imports_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("example", "example_project.extension"),
    )
    imports: list[str] = []
    monkeypatch.setattr(plugin_runtime, "import_module", imports.append)
    plan = prepare_extension_plugins(_config(plugins=["example"]))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: activate_extension_plugins(plan), range(2)))

    assert imports == ["example_project.extension"]
    assert results[0].provenance == results[1].provenance


def test_provenance_parser_is_strict_stable_and_checkpoint_safe() -> None:
    parsed = parse_extension_plugin_provenance(
        [
            {
                "name": "zeta",
                "distribution": "zeta-project",
                "version": "2.0",
                "target": "zeta.extension",
            },
            {
                "name": "alpha",
                "distribution": "alpha-project",
                "version": "1.0",
                "target": "alpha.extension",
            },
        ]
    )

    assert [item.name for item in parsed] == ["alpha", "zeta"]
    assert parse_extension_plugin_provenance(
        extension_plugin_provenance_to_dicts(parsed)
    ) == parsed

    with pytest.raises(ExtensionIdentityError, match="invalid fields"):
        parse_extension_plugin_provenance(
            [
                {
                    "name": "alpha",
                    "distribution": "alpha-project",
                    "version": "1.0",
                    "target": "alpha.extension",
                    "extra": True,
                }
            ]
        )
    with pytest.raises(ExtensionIdentityError, match="duplicate"):
        parse_extension_plugin_provenance(
            extension_plugin_provenance_to_dicts(parsed[:1]) * 2
        )


def test_prepare_and_resolve_use_immutable_config_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        FakeEntryPoint("example", "example_project.extension"),
    )
    monkeypatch.setattr(plugin_runtime, "import_module", lambda _: None)
    config = _config(plugins=["example"])
    config.data.params["shape"] = (1, 2)
    config.model.params["dtype"] = torch.float64
    plan = prepare_extension_plugins(config)
    different_config = _config(plugins=["example"])
    different_config.data.params["shape"] = (9, 9)
    assert plan != prepare_extension_plugins(different_config)
    config.model.params["changed"] = True
    detached_config = plan.config
    detached_config.trainer.num_epochs = 0
    detached_config.extensions.plugins = ["other"]
    with pytest.raises(ValueError, match=r"trainer\.num_epochs must be positive"):
        detached_config.validate()

    resolved = activate_extension_plugins(plan)

    assert plan.config is not detached_config
    assert plan.config.trainer.num_epochs == 1
    assert "changed" not in resolved.config.model.params
    assert resolved.config.data.params["shape"] == (1, 2)
    assert type(resolved.config.data.params["shape"]) is tuple
    assert resolved.config.model.params["dtype"] is torch.float64
    assert resolved.config.trainer.num_epochs == 1
    assert resolved.config.extensions.plugins == ["example"]

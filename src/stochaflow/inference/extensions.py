"""Shared extension selection for checkpoint-backed inference operations."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from stochaflow.utils.config import ConfigError, StochaflowConfig
from stochaflow.utils.plugins import (
    ExtensionActivationPlan,
    ExtensionIdentityError,
    ExtensionPluginProvenance,
    ExtensionSelectionPolicy,
    prepare_extension_plugins,
)


def merge_checkpoint_extension_config(
    checkpoint: StochaflowConfig,
    *,
    additions: tuple[str, ...],
    expected_plugin_names: tuple[str, ...],
) -> tuple[StochaflowConfig, bool]:
    """Merge additive invocation plugins without weakening checkpoint identity."""

    config = deepcopy(checkpoint)
    added_plugins = False
    resolved_plugins = list(expected_plugin_names)
    checkpoint_plugins = checkpoint.extensions.plugins
    if checkpoint_plugins is not None:
        missing = sorted(set(expected_plugin_names) - set(checkpoint_plugins))
        unexpected = sorted(set(checkpoint_plugins) - set(expected_plugin_names))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(
                    "missing provenance-required plugin(s): " + ", ".join(missing)
                )
            if unexpected:
                details.append(
                    "unproven config-only plugin(s): " + ", ".join(unexpected)
                )
            raise ExtensionIdentityError(
                "checkpoint config extension selection conflicts with checkpoint "
                "provenance: " + "; ".join(details)
            )
    for plugin in additions:
        if plugin not in resolved_plugins:
            resolved_plugins.append(plugin)
            added_plugins = True
    config.extensions.plugins = resolved_plugins
    config.validate()
    selected = config.extensions.plugins
    if selected is not None:
        missing = sorted(set(expected_plugin_names) - set(selected))
        if missing:
            raise ConfigError(
                "invocation config cannot remove checkpoint-required extension "
                "plugin(s): " + ", ".join(missing)
            )
    return config, added_plugins


def prepare_checkpoint_extension_plan(
    checkpoint: StochaflowConfig,
    *,
    additions: tuple[str, ...],
    expected_provenance: tuple[ExtensionPluginProvenance, ...],
    plan_factory: Callable[..., ExtensionActivationPlan] = prepare_extension_plugins,
) -> ExtensionActivationPlan:
    """Prepare exact checkpoint plugins plus explicitly additive extensions."""

    expected_names = tuple(item.name for item in expected_provenance)
    activation_config, added_plugins = merge_checkpoint_extension_config(
        checkpoint,
        additions=additions,
        expected_plugin_names=expected_names,
    )
    plan = plan_factory(
        activation_config,
        expected_provenance=expected_provenance,
        selection_policy=(
            ExtensionSelectionPolicy.INTERSECTION
            if added_plugins
            else ExtensionSelectionPolicy.EXACT
        ),
    )
    selected_plugin_names = {item.name for item in plan.provenance}
    missing_required_plugins = sorted(set(expected_names) - selected_plugin_names)
    if missing_required_plugins:
        raise ExtensionIdentityError(
            "invocation config is missing checkpoint-required extension plugin(s): "
            + ", ".join(missing_required_plugins)
        )
    return plan


__all__ = [
    "merge_checkpoint_extension_config",
    "prepare_checkpoint_extension_plan",
]

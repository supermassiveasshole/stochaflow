"""Tests for CLI-only extension version confirmation policy."""

from io import StringIO

from stochaflow.scripts import extensions_cli
from stochaflow.utils.config import load_config_dict
from stochaflow.utils.plugins import (
    ExtensionActivationPlan,
    ExtensionPluginProvenance,
    ExtensionSelectionPolicy,
    ExtensionVersionMismatch,
    ExtensionVersionPolicy,
    ResolvedExtensions,
)


class FixtureTTYInput(StringIO):
    def isatty(self) -> bool:
        return True


def _plan() -> ExtensionActivationPlan:
    config = load_config_dict(
        {
            "experiment": {"name": "test"},
            "data": {"name": "data"},
            "model": {"name": "model"},
            "training": {"name": "training"},
            "extensions": {"plugins": ["example"]},
        }
    )
    expected = ExtensionPluginProvenance("example", "example", "1.0", "example")
    current = ExtensionPluginProvenance("example", "example", "2.0", "example")
    return ExtensionActivationPlan(
        config,
        (current,),
        (ExtensionVersionMismatch(expected, current),),
        ExtensionSelectionPolicy.EXACT,
    )


def _record_activation(monkeypatch):
    observed = {}

    def activate(plan, *, policy=ExtensionVersionPolicy.REJECT, acceptance_method=None):
        observed.update(policy=policy, acceptance_method=acceptance_method)
        return ResolvedExtensions(plan.config, plan.provenance, ())

    monkeypatch.setattr(extensions_cli, "activate_extension_plugins", activate)
    return observed


def test_force_flag_accepts_version_mismatch_without_prompt(monkeypatch):
    observed = _record_activation(monkeypatch)

    extensions_cli.activate_extensions_for_cli(
        _plan(),
        force_version_mismatch=True,
    )

    assert observed == {
        "policy": ExtensionVersionPolicy.ALLOW,
        "acceptance_method": "force-flag",
    }


def test_interactive_confirmation_records_prompt(monkeypatch):
    observed = _record_activation(monkeypatch)
    monkeypatch.setattr(extensions_cli.sys, "stdin", FixtureTTYInput(""))
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")

    extensions_cli.activate_extensions_for_cli(
        _plan(),
        force_version_mismatch=False,
    )

    assert observed == {
        "policy": ExtensionVersionPolicy.ALLOW,
        "acceptance_method": "prompt",
    }


def test_noninteractive_invocation_uses_reject_policy(monkeypatch):
    observed = _record_activation(monkeypatch)
    monkeypatch.setattr(extensions_cli.sys, "stdin", StringIO(""))

    extensions_cli.activate_extensions_for_cli(
        _plan(),
        force_version_mismatch=False,
    )

    assert observed == {
        "policy": ExtensionVersionPolicy.REJECT,
        "acceptance_method": None,
    }

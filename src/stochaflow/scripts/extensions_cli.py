"""CLI-only policy for extension version mismatch confirmation."""

from __future__ import annotations

import sys

from stochaflow.utils.plugins import (
    ExtensionActivationPlan,
    ExtensionVersionPolicy,
    ResolvedExtensions,
    activate_extension_plugins,
)


def activate_extensions_for_cli(
    plan: ExtensionActivationPlan,
    *,
    force_version_mismatch: bool,
) -> ResolvedExtensions:
    """Activate a preflighted plan using explicit CLI confirmation semantics."""

    if not plan.version_mismatches:
        return activate_extension_plugins(plan)

    print(
        "Warning: installed extension plugin versions differ from the checkpoint:",
        file=sys.stderr,
    )
    for mismatch in plan.version_mismatches:
        print(
            f"  {mismatch.current.name}: checkpoint={mismatch.expected.version}, "
            f"installed={mismatch.current.version}",
            file=sys.stderr,
        )
    if force_version_mismatch:
        return activate_extension_plugins(
            plan,
            policy=ExtensionVersionPolicy.ALLOW,
            acceptance_method="force-flag",
        )
    if sys.stdin.isatty():
        response = input("Continue with the installed extension versions? [y/N] ")
        if response.strip().lower() in {"y", "yes"}:
            return activate_extension_plugins(
                plan,
                policy=ExtensionVersionPolicy.ALLOW,
                acceptance_method="prompt",
            )
    return activate_extension_plugins(plan)


__all__ = ["activate_extensions_for_cli"]

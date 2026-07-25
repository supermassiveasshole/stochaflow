"""Utility primitives.

Heavy runtime builders live in ``stochaflow.utils.factory`` and should be
imported from that module directly to avoid package-level import cycles.
"""

from .checkpoint import CheckpointManager, CheckpointState, LoadedCheckpoint
from .config import (
    ComponentConfig,
    ConfigError,
    EarlyStoppingConfig,
    EMAConfig,
    LoggingConfig,
    LRSchedulerConfig,
    StochaflowConfig,
    coerce_config_section,
    load_config,
    load_config_dict,
)
from .logging import ExperimentLogger
from .plugins import (
    EXTENSION_ENTRY_POINT_GROUP,
    ExtensionActivationError,
    ExtensionActivationPlan,
    ExtensionActivationStateError,
    ExtensionDiscoveryError,
    ExtensionIdentityError,
    ExtensionPluginError,
    ExtensionPluginProvenance,
    ExtensionSelectionPolicy,
    ExtensionVersionAcceptance,
    ExtensionVersionMismatch,
    ExtensionVersionMismatchError,
    ExtensionVersionPolicy,
    ResolvedExtensions,
    activate_extension_plugins,
    extension_plugin_provenance_to_dicts,
    parse_extension_plugin_provenance,
    prepare_extension_plugins,
)
from .registry import REGISTRIES, Registry, RegistryCatalog, RegistryError

__all__ = [
    "EXTENSION_ENTRY_POINT_GROUP",
    "REGISTRIES",
    "CheckpointManager",
    "CheckpointState",
    "ComponentConfig",
    "ConfigError",
    "EMAConfig",
    "EarlyStoppingConfig",
    "ExperimentLogger",
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
    "LRSchedulerConfig",
    "LoadedCheckpoint",
    "LoggingConfig",
    "Registry",
    "RegistryCatalog",
    "RegistryError",
    "ResolvedExtensions",
    "StochaflowConfig",
    "activate_extension_plugins",
    "coerce_config_section",
    "extension_plugin_provenance_to_dicts",
    "load_config",
    "load_config_dict",
    "parse_extension_plugin_provenance",
    "prepare_extension_plugins",
]

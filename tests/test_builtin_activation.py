"""Built-in activation ownership and operation import-boundary contracts."""

from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from inspect import signature
from pathlib import Path
from typing import get_type_hints

import pytest

from stochaflow import _builtin_activation as activation
from stochaflow.extensions import ExperimentLogger as ExtensionExperimentLogger
from stochaflow.utils import ExperimentLogger as UtilityExperimentLogger
from stochaflow.utils import factory as legacy_factory
from stochaflow.utils.logging import (
    CompositeLogger,
    ExperimentLogger,
    NullLogger,
)

REPOSITORY_ROOT = Path(__file__).parents[1]


def _run_fresh_process(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_operation_scopes_have_fixed_minimal_module_order() -> None:
    assert activation.SAMPLING_BUILTIN_MODULES == (
        "stochaflow.models",
        "stochaflow.processes",
        "stochaflow.sampling",
    )
    assert activation.EVALUATION_BUILTIN_MODULES == (
        "stochaflow.data",
        "stochaflow.metrics.builtin",
        "stochaflow.metrics.reference",
        "stochaflow.models",
        "stochaflow.processes",
        "stochaflow.sampling",
    )
    assert activation.TRAINING_BUILTIN_MODULES == (
        "stochaflow.data",
        "stochaflow.metrics.builtin",
        "stochaflow.metrics.reference",
        "stochaflow.models",
        "stochaflow.processes",
        "stochaflow.sampling",
        "stochaflow.training",
        "stochaflow.training.diagnostics",
        "stochaflow.utils.logging",
    )


def test_activation_is_ordered_idempotent_and_overlap_safe(
) -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from stochaflow import _builtin_activation as activation",
                "imports = []",
                "activation.import_module = imports.append",
                "activation._activate_modules(('first', 'shared'))",
                "activation._activate_modules(('shared', 'last'))",
                "activation._activate_modules(('first', 'shared', 'last'))",
                "assert imports == ['first', 'shared', 'last']",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_concurrent_first_activation_imports_each_module_once(
) -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import time",
                "from concurrent.futures import ThreadPoolExecutor",
                "from stochaflow import _builtin_activation as activation",
                "imports = []",
                "def record_import(module_name):",
                "    imports.append(module_name)",
                "    time.sleep(0.01)",
                "activation.import_module = record_import",
                "with ThreadPoolExecutor(max_workers=2) as executor:",
                (
                    "    results = tuple(executor.map(lambda _: "
                    "activation._activate_modules(('one', 'two')), range(2)))"
                ),
                "assert results == (None, None)",
                "assert imports == ['one', 'two']",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_reentry_poisons_activation_and_preserves_first_cause(
) -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from stochaflow import _builtin_activation as activation",
                "def reenter(_):",
                "    activation._activate_modules(('nested',))",
                "activation.import_module = reenter",
                "try:",
                "    activation._activate_modules(('outer',))",
                "except RuntimeError as first:",
                "    assert 're-entered' in str(first)",
                "    original_cause = first.__cause__",
                "    assert isinstance(original_cause, RuntimeError)",
                "    assert 're-entered' in str(original_cause)",
                "else:",
                "    raise AssertionError('reentry unexpectedly succeeded')",
                "try:",
                "    activation._activate_modules(('another',))",
                "except RuntimeError as later:",
                "    assert 'restart' in str(later)",
                "    assert later.__cause__ is original_cause",
                "else:",
                "    raise AssertionError('poisoned activation succeeded')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_swallowed_reentry_still_prevents_module_completion() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from types import ModuleType",
                "from stochaflow import _builtin_activation as activation",
                "def swallow_reentry(_):",
                "    try:",
                "        activation._activate_modules(('nested',))",
                "    except RuntimeError:",
                "        pass",
                "    return ModuleType('outer')",
                "activation.import_module = swallow_reentry",
                "try:",
                "    activation._activate_modules(('outer',))",
                "except RuntimeError as error:",
                "    assert 'previously failed' in str(error)",
                "else:",
                "    raise AssertionError('swallowed reentry escaped poison')",
                "assert 'outer' not in activation._activation_runtime.completed_modules",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_reentry_then_different_failure_still_reports_first_cause() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from stochaflow import _builtin_activation as activation",
                "def fail_after_reentry(_):",
                "    try:",
                "        activation._activate_modules(('nested',))",
                "    except RuntimeError:",
                "        pass",
                "    raise LookupError('later import failure')",
                "activation.import_module = fail_after_reentry",
                "try:",
                "    activation._activate_modules(('outer',))",
                "except RuntimeError as error:",
                "    assert 'previously failed' in str(error)",
                "    assert isinstance(error.__cause__, RuntimeError)",
                "    assert 're-entered' in str(error.__cause__)",
                "else:",
                "    raise AssertionError('later error replaced activation poison')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_reentry_then_finally_failure_still_reports_first_cause() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from stochaflow import _builtin_activation as activation",
                "def fail_in_finally(_):",
                "    try:",
                "        activation._activate_modules(('nested',))",
                "    finally:",
                "        raise LookupError('later import failure')",
                "activation.import_module = fail_in_finally",
                "try:",
                "    activation._activate_modules(('outer',))",
                "except RuntimeError as error:",
                "    assert 'previously failed' in str(error)",
                "    assert isinstance(error.__cause__, RuntimeError)",
                "    assert 're-entered' in str(error.__cause__)",
                "else:",
                "    raise AssertionError('finally error replaced activation poison')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "failure_expression",
    [
        "RegistryError('model registrations must inherit nn.Module')",
        "RegistryError(\"model 'unet' already registered\")",
        "ImportError('broken built-in')",
    ],
)
def test_import_registration_failure_poisons_process(
    failure_expression: str,
) -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from stochaflow import _builtin_activation as activation",
                "from stochaflow.utils.registry import RegistryError",
                f"failure = {failure_expression}",
                "def fail(_):",
                "    raise failure",
                "activation.import_module = fail",
                "try:",
                "    activation._activate_modules(('broken',))",
                "except RuntimeError as first:",
                "    assert 'restart' in str(first)",
                "    assert first.__cause__ is failure",
                "else:",
                "    raise AssertionError('broken activation succeeded')",
                "try:",
                "    activation._activate_modules(('unused',))",
                "except RuntimeError as later:",
                "    assert 'previously failed' in str(later)",
                "    assert later.__cause__ is failure",
                "else:",
                "    raise AssertionError('poisoned activation succeeded')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_real_wrong_base_failure_poisons_data_activation() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from stochaflow._builtin_activation import activate_data_builtins",
                "from stochaflow.utils.registry import REGISTRIES, RegistryError",
                "REGISTRIES.data_builders.add('invalid', object)",
                "try:",
                "    activate_data_builtins()",
                "except RuntimeError as first:",
                "    cause = first.__cause__",
                "    assert isinstance(cause, RegistryError)",
                "    assert 'do not inherit DataBuilder' in str(cause)",
                "else:",
                "    raise AssertionError('wrong-base activation succeeded')",
                "try:",
                "    activate_data_builtins()",
                "except RuntimeError as later:",
                "    assert later.__cause__ is cause",
                "    assert 'previously failed' in str(later)",
                "else:",
                "    raise AssertionError('poisoned activation succeeded')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_real_duplicate_failure_keeps_prior_registrations_without_rollback() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from torch import nn",
                (
                    "from stochaflow._builtin_activation import "
                    "activate_model_builtins"
                ),
                "from stochaflow.utils.registry import REGISTRIES, RegistryError",
                "class PlaceholderUNet(nn.Module):",
                "    pass",
                "REGISTRIES.models.add('unet', PlaceholderUNet)",
                "try:",
                "    activate_model_builtins()",
                "except RuntimeError as first:",
                "    cause = first.__cause__",
                "    assert isinstance(cause, RegistryError)",
                "    assert \"model 'unet' already registered\" in str(cause)",
                "else:",
                "    raise AssertionError('duplicate activation succeeded')",
                "assert REGISTRIES.models.resolve('unet') is PlaceholderUNet",
                "assert REGISTRIES.models.names() == ('adm_unet', 'dit', 'unet')",
                "try:",
                "    activate_model_builtins()",
                "except RuntimeError as later:",
                "    assert later.__cause__ is cause",
                "else:",
                "    raise AssertionError('poisoned activation succeeded')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_resolved_execution_check_never_imports_modules(
) -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from stochaflow import _builtin_activation as activation",
                "def unexpected_import(_):",
                "    raise AssertionError('resolved execution imported built-ins')",
                "activation.import_module = unexpected_import",
                "try:",
                "    activation.require_sampling_builtins()",
                "except RuntimeError as error:",
                "    assert 'not activated' in str(error)",
                "else:",
                "    raise AssertionError('missing activation was accepted')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_fresh_operation_scopes_exclude_training_and_logger_registrations() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import sys",
                (
                    "from stochaflow._builtin_activation import "
                    "activate_sampling_builtins, activate_evaluation_builtins, "
                    "activate_training_builtins"
                ),
                "from stochaflow.utils.registry import REGISTRIES, RegistryError",
                "def snapshot():",
                "    return {",
                "        name: getattr(REGISTRIES, name).names()",
                "        for name in (",
                "            'models', 'data_builders',",
                "            'sampling_artifact_writers', 'noise_schedules',",
                "            'processes', 'samplers', 'sampling_builders',",
                "            'training_builders', 'evaluation_builders',",
                "            'objectives', 'metrics', 'optimizers',",
                "            'lr_schedulers', 'loggers', 'diagnostics'",
                "        )",
                "    }",
                "activate_sampling_builtins()",
                "assert snapshot() == {",
                "    'models': ('adm_unet', 'dit', 'unet'),",
                "    'data_builders': (),",
                "    'sampling_artifact_writers': ('image', 'tensor'),",
                "    'noise_schedules': ('cosine_alpha_bar', 'linear_beta'),",
                "    'processes': ('discrete_gaussian',),",
                "    'samplers': ('ddim', 'ddpm'),",
                (
                    "    'sampling_builders': "
                    "('class_conditional_denoising', 'standard_denoising'),"
                ),
                "    'training_builders': (),",
                "    'evaluation_builders': (),",
                "    'objectives': (),",
                "    'metrics': (),",
                "    'optimizers': (),",
                "    'lr_schedulers': (),",
                "    'loggers': (),",
                "    'diagnostics': (),",
                "}",
                "assert 'stochaflow.training.trainer' not in sys.modules",
                (
                    "assert not any(name == 'stochaflow.training.diagnostics' or "
                    "name.startswith('stochaflow.training.diagnostics.') "
                    "for name in sys.modules)"
                ),
                "activate_evaluation_builtins()",
                "evaluation = snapshot()",
                (
                    "assert evaluation['data_builders'] == "
                    "('class_labeled_image', 'image', "
                    "'multi_resolution_image', 'super_resolution')"
                ),
                "assert evaluation['metrics'] == ('fid', 'kid', 'mae', 'mean', 'mse')",
                "assert evaluation['training_builders'] == ()",
                "assert evaluation['evaluation_builders'] == ()",
                "assert evaluation['objectives'] == ()",
                "assert evaluation['lr_schedulers'] == ()",
                "assert evaluation['loggers'] == ()",
                "assert evaluation['diagnostics'] == ()",
                "from stochaflow.data import IMAGE_DATA_SOURCES",
                (
                    "assert IMAGE_DATA_SOURCES.names() == "
                    "('image_folder', 'paired_image_folders', 'torchvision')"
                ),
                "assert 'stochaflow.training.trainer' not in sys.modules",
                "try:",
                "    from torch import nn",
                "    REGISTRIES.models.add('unet', nn.Linear)",
                "except RegistryError:",
                "    pass",
                "else:",
                "    raise AssertionError('built-in registration must win')",
                "activate_training_builtins()",
                "training = snapshot()",
                (
                    "assert training['training_builders'] == "
                    "('class_conditional_gaussian_denoising', "
                    "'gaussian_denoising', 'supervised')"
                ),
                "assert training['objectives'] == ('mse',)",
                "assert training['lr_schedulers'] == ('warmup_cosine',)",
                "assert training['loggers'] == ('local', 'tensorboard', 'wandb')",
                (
                    "assert training['diagnostics'] == "
                    "('class_conditional_diffusion_quality', 'diffusion_quality')"
                ),
                "assert training['evaluation_builders'] == ()",
                "assert training['optimizers'] == ()",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_facade_imports_do_not_register_framework_owned_builtins() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import stochaflow.data",
                "import stochaflow.metrics.builtin",
                "import stochaflow.metrics.reference",
                "import stochaflow.models",
                "import stochaflow.processes",
                "import stochaflow.sampling",
                "import stochaflow.training",
                "import stochaflow.training.diagnostics",
                "import stochaflow.utils.logging",
                "from stochaflow.data import IMAGE_DATA_SOURCES",
                "from stochaflow.training.diagnostics import DIAGNOSTIC_PROVIDERS",
                "from stochaflow.utils.registry import REGISTRIES",
                (
                    "assert all(not getattr(REGISTRIES, name).names() for name in "
                    "('models', 'data_builders', 'sampling_artifact_writers', "
                    "'noise_schedules', 'processes', 'samplers', "
                    "'sampling_builders', 'training_builders', 'objectives', "
                    "'metrics', 'lr_schedulers', 'loggers', 'diagnostics'))"
                ),
                "assert IMAGE_DATA_SOURCES.names() == ()",
                (
                    "assert all(not DIAGNOSTIC_PROVIDERS.registry(name).names() "
                    "for name in ('step_metrics', 'sampler_metrics', "
                    "'denoiser_artifacts', 'sampler_artifacts', "
                    "'reference_metrics'))"
                ),
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_parser_keeps_sample_and_evaluate_import_closure_training_free() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import sys",
                "from stochaflow.scripts.cli import build_argument_parser",
                "build_argument_parser()",
                "from stochaflow.utils.registry import REGISTRIES",
                "assert 'stochaflow.training.trainer' not in sys.modules",
                (
                    "assert not any(name == 'stochaflow.training.diagnostics' or "
                    "name.startswith('stochaflow.training.diagnostics.') "
                    "for name in sys.modules)"
                ),
                "assert REGISTRIES.loggers.names() == ()",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_factory_import_does_not_activate_builtins() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import stochaflow._component_factory",
                "import stochaflow.utils.factory",
                "from stochaflow.utils.registry import REGISTRIES",
                "assert REGISTRIES.models.names() == ()",
                "assert REGISTRIES.processes.names() == ()",
                "assert REGISTRIES.training_builders.names() == ()",
                "assert REGISTRIES.loggers.names() == ()",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_modules_do_not_import_legacy_factory() -> None:
    source_root = REPOSITORY_ROOT / "src" / "stochaflow"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        if path == source_root / "utils" / "factory.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "stochaflow.utils.factory"
                for alias in node.names
            ):
                violations.append(str(path.relative_to(REPOSITORY_ROOT)))
            if isinstance(node, ast.ImportFrom):
                direct = node.module == "stochaflow.utils.factory"
                package_member = (
                    node.module == "stochaflow.utils"
                    and any(alias.name == "factory" for alias in node.names)
                )
                if direct or package_member:
                    violations.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert violations == []


def test_default_inference_helpers_activate_narrow_model_and_process_scopes() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import torch",
                (
                    "from stochaflow.inference.checkpoint import "
                    "build_checkpointed_process, build_inference_model_provider"
                ),
                "from stochaflow.utils.config import load_config_dict",
                "from stochaflow.utils.registry import REGISTRIES",
                "config = load_config_dict({",
                "    'experiment': {'name': 'direct-inference'},",
                "    'data': {'name': 'unused.data', 'params': {}},",
                "    'model': {",
                "        'name': 'unet',",
                "        'params': {",
                "            'in_channels': 1, 'out_channels': 1,",
                "            'base_channels': 32, 'channel_multipliers': [1],",
                "            'num_res_blocks': 1, 'time_embedding_dim': 32,",
                "            'attention_heads': 1",
                "        },",
                "    },",
                "    'process': {",
                "        'name': 'discrete_gaussian',",
                "        'params': {",
                "            'schedule': {",
                "                'name': 'linear_beta',",
                "                'params': {'num_timesteps': 2}",
                "            }",
                "        },",
                "    },",
                "    'training': {'name': 'unused.training', 'params': {}},",
                "    'trainer': {'precision': 'fp32'},",
                "})",
                "assert REGISTRIES.models.names() == ()",
                "provider = build_inference_model_provider(",
                "    config, {'model_state_dict': {}}, device=torch.device('cpu')",
                ")",
                "assert REGISTRIES.models.names() == ('adm_unet', 'dit', 'unet')",
                "try:",
                "    provider.resolve('raw')",
                "except ValueError as error:",
                "    assert 'state' in str(error)",
                "else:",
                "    raise AssertionError('empty model state unexpectedly loaded')",
                "assert REGISTRIES.processes.names() == ()",
                "try:",
                "    build_checkpointed_process(",
                "        config, {'process_state_dict': {}},",
                "        device=torch.device('cpu')",
                "    )",
                "except ValueError as error:",
                "    assert 'state' in str(error)",
                "else:",
                "    raise AssertionError('empty process state unexpectedly loaded')",
                "assert REGISTRIES.processes.names() == ('discrete_gaussian',)",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_default_inference_asset_helper_activates_models_before_lazy_get() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import torch",
                (
                    "from stochaflow.inference.checkpoint import "
                    "build_inference_asset_provider"
                ),
                "from stochaflow.utils.registry import REGISTRIES",
                "payload = {",
                "    'inference_asset_descriptors': {",
                "        'codec': {",
                "            'training_asset_name': 'codec_module',",
                "            'declaration': {",
                "                'name': 'unet',",
                "                'params': {",
                "                    'in_channels': 1, 'out_channels': 1,",
                "                    'base_channels': 32,",
                "                    'channel_multipliers': [1],",
                "                    'num_res_blocks': 1,",
                "                    'time_embedding_dim': 32,",
                "                    'attention_heads': 1",
                "                }",
                "            },",
                "            'capability_role': 'image_codec',",
                "            'persistence': 'embedded_state'",
                "        }",
                "    },",
                "    'inference_asset_state_dicts': {'codec_module': {}},",
                "}",
                "assert REGISTRIES.models.names() == ()",
                "provider = build_inference_asset_provider(",
                "    payload, device=torch.device('cpu')",
                ")",
                "assert REGISTRIES.models.names() == ('adm_unet', 'dit', 'unet')",
                "try:",
                "    provider.get('codec', expected_capability_role='image_codec')",
                "except ValueError as error:",
                "    assert 'state' in str(error)",
                "else:",
                "    raise AssertionError('empty asset state unexpectedly loaded')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_custom_inference_model_factory_does_not_activate_global_models() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from types import SimpleNamespace",
                "import torch",
                "from torch import nn",
                (
                    "from stochaflow.inference.checkpoint import "
                    "build_inference_model_provider"
                ),
                "from stochaflow.utils.config import ComponentConfig",
                "from stochaflow.utils.registry import REGISTRIES",
                "state = nn.Linear(1, 1).state_dict()",
                "config = SimpleNamespace(",
                "    model=ComponentConfig(name='custom', params={})",
                ")",
                "provider = build_inference_model_provider(",
                "    config, {'model_state_dict': state},",
                "    device=torch.device('cpu'),",
                "    model_factory=lambda _: nn.Linear(1, 1),",
                ")",
                "provider.resolve('raw')",
                "assert REGISTRIES.models.names() == ()",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_injected_metric_and_sampling_registries_do_not_activate_globals() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import torch",
                "from torch import nn",
                "from torchmetrics import Metric",
                "from stochaflow.inference import PinnedInferenceModelProvider",
                "from stochaflow.metrics import MetricEngine, MetricSpec",
                "from stochaflow.sampling import (",
                "    SamplingBatch, SamplingBuilder, SamplingBuilderContext,",
                "    SamplingOutput, execute_sampling_builder",
                ")",
                "from stochaflow.utils.registry import REGISTRIES, Registry",
                "class CustomMetric(Metric):",
                "    def __init__(self):",
                "        super().__init__()",
                "        self.add_state('value', default=torch.tensor(0.0))",
                "    def update(self, value):",
                "        self.value += value",
                "    def compute(self):",
                "        return self.value",
                "metrics = Registry('custom metric', expected_type=Metric)",
                "metrics.add('custom', CustomMetric)",
                "MetricEngine([",
                "    MetricSpec(id='value', name='custom', channel='value')",
                "], registry=metrics)",
                "assert REGISTRIES.metrics.names() == ()",
                "class CustomBuilder(SamplingBuilder):",
                "    def run(self):",
                "        return SamplingOutput(",
                "            (SamplingBatch(torch.ones(1, 1), num_samples=1),), {}",
                "        )",
                (
                    "builders = Registry('custom sampling builder', "
                    "expected_type=SamplingBuilder)"
                ),
                "builders.add('custom', CustomBuilder)",
                "model = nn.Identity()",
                "model.eval()",
                "provider = PinnedInferenceModelProvider(",
                "    model=model, weights='raw', device=torch.device('cpu')",
                ")",
                "context = SamplingBuilderContext(",
                "    params={}, process=None, model_provider=provider,",
                "    device=torch.device('cpu'), seed=1, shape=None,",
                "    num_samples=1, batch_size=1",
                ")",
                "execute_sampling_builder('custom', context, registry=builders)",
                "assert REGISTRIES.sampling_builders.names() == ()",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_builtin_diagnostic_activates_providers_and_sampler_pool() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "import torch",
                (
                    "from stochaflow.training.diagnostics import "
                    "DIAGNOSTIC_PROVIDERS, DiffusionQualityDiagnostic"
                ),
                "from stochaflow.training.diagnostics.config import (",
                "    SamplerProfileConfig, TrajectoryProviderConfig",
                ")",
                (
                    "from stochaflow.training.diagnostics.runtime import "
                    "ClassConditionalSamplerPool"
                ),
                "from stochaflow.utils.logging import NullLogger",
                "from stochaflow.utils.registry import REGISTRIES",
                "assert DIAGNOSTIC_PROVIDERS.step_metrics.names() == ()",
                "DiffusionQualityDiagnostic(",
                "    logger=NullLogger(), output_dir='unused',",
                "    samplers=[{'id': 'ddpm', 'name': 'ddpm'}],",
                "    sampling={'shape': [1, 4, 4]},",
                ")",
                "assert DIAGNOSTIC_PROVIDERS.step_metrics.names() == (",
                "    'noise_alignment', 'timestep_bucket_loss',",
                "    'x0_reconstruction'",
                ")",
                "assert REGISTRIES.samplers.names() == ()",
                "profile = SamplerProfileConfig(",
                "    id='ddpm', name='ddpm', params={},",
                "    trajectory=TrajectoryProviderConfig()",
                ")",
                "ClassConditionalSamplerPool([profile], device=torch.device('cpu'))",
                "assert REGISTRIES.samplers.names() == ('ddim', 'ddpm')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_logger_contract_reexports_preserve_identity_and_pickle_location() -> None:
    assert ExtensionExperimentLogger is ExperimentLogger
    assert UtilityExperimentLogger is ExperimentLogger
    for logger_type in (ExperimentLogger, NullLogger, CompositeLogger):
        assert logger_type.__module__ == "stochaflow.utils.logging"
        assert pickle.loads(pickle.dumps(logger_type)) is logger_type


def test_logger_contract_installs_base_without_registering_backends() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from stochaflow import extensions",
                "from stochaflow.utils.registry import REGISTRIES, RegistryError",
                "assert REGISTRIES.loggers.names() == ()",
                "try:",
                "    REGISTRIES.loggers.add('invalid', object)",
                "except RegistryError as error:",
                "    assert 'ExperimentLogger' in str(error)",
                "else:",
                "    raise AssertionError('logger base contract was not installed')",
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_factory_forwards_have_resolvable_annotations() -> None:
    TrainingComponents = legacy_factory.TrainingComponents
    functions = (
        legacy_factory.build_process,
        legacy_factory.build_diagnostics,
        legacy_factory.build_ema,
        legacy_factory.build_training_components,
    )
    for function in functions:
        assert get_type_hints(function)
        assert tuple(signature(function).parameters)

    hints = get_type_hints(TrainingComponents)
    assert hints["process"]
    assert TrainingComponents.__module__ == "stochaflow.utils.factory"
    assert pickle.loads(pickle.dumps(TrainingComponents)) is TrainingComponents
    assert dir(legacy_factory).count("TrainingComponents") == 1


def test_legacy_factory_annotations_resolve_in_arbitrary_order() -> None:
    result = _run_fresh_process(
        "\n".join(
            (
                "from typing import get_type_hints",
                "import stochaflow.utils.factory as factory",
                "for function in (",
                "    factory.build_process, factory.build_diagnostics,",
                "    factory.build_ema, factory.build_training_components",
                "):",
                "    assert get_type_hints(function)",
                "assert get_type_hints(factory.TrainingComponents)",
                (
                    "assert factory.TrainingComponents.__module__ == "
                    "'stochaflow.utils.factory'"
                ),
            )
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr

"""Tests for lazy checkpoint-embedded inference asset reconstruction."""

from collections.abc import Mapping
from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.inference import InferenceAssetProvider
from stochaflow.utils.checkpoint import InferenceAssetDescriptor
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.factory import build_model
from stochaflow.utils.registry import RegistryError


class InferenceAssetModule(nn.Module):
    """Small parameter-bearing inference asset used by provider tests."""

    def __init__(self, *, width: int = 2) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(width))


def _descriptors(
    *,
    slot: str = "codec",
    training_asset_name: str = "training_codec",
    component_name: str = "test_codec",
    role: str = "image_codec",
    width: int = 2,
) -> dict[str, InferenceAssetDescriptor]:
    return {
        slot: {
            "training_asset_name": training_asset_name,
            "declaration": {
                "name": component_name,
                "params": {"width": width},
            },
            "capability_role": role,
            "persistence": "embedded_state",
        }
    }


def _state(*, width: int = 2, value: float = 3.0) -> dict[str, torch.Tensor]:
    module = InferenceAssetModule(width=width)
    with torch.no_grad():
        module.scale.fill_(value)
    return dict(module.state_dict())


def test_provider_is_lazy_requested_only_and_caches_success() -> None:
    calls: list[ComponentConfig] = []
    descriptors = {
        **_descriptors(),
        **_descriptors(
            slot="conditioner",
            training_asset_name="training_conditioner",
            component_name="test_conditioner",
            role="conditioner",
        ),
    }

    def model_factory(component: ComponentConfig) -> nn.Module:
        calls.append(component)
        component.params["width"] = int(component.params["width"])
        return InferenceAssetModule(width=component.params["width"])

    provider = InferenceAssetProvider(
        descriptors=descriptors,
        state_dicts={
            "training_codec": _state(value=4.0),
            "training_conditioner": _state(value=9.0),
        },
        device=torch.device("cpu"),
        model_factory=model_factory,
    )

    assert calls == []
    resolved = provider.get("codec", expected_capability_role="image_codec")

    assert isinstance(resolved, InferenceAssetModule)
    assert resolved.training is False
    assert resolved.scale.device.type == "cpu"
    assert torch.equal(resolved.scale, torch.full((2,), 4.0))
    assert provider.get(
        "codec", expected_capability_role="image_codec"
    ) is resolved
    assert [component.name for component in calls] == ["test_codec"]
    assert descriptors["codec"]["declaration"]["params"] == {"width": 2}


def test_provider_rejects_unknown_slot_and_wrong_role_before_factory() -> None:
    factory_calls = 0

    def model_factory(component: ComponentConfig) -> nn.Module:
        nonlocal factory_calls
        factory_calls += 1
        return InferenceAssetModule(**component.params)

    provider = InferenceAssetProvider(
        descriptors=_descriptors(),
        state_dicts={"training_codec": _state()},
        device=torch.device("cpu"),
        model_factory=model_factory,
    )

    with pytest.raises(KeyError, match="unknown inference asset slot"):
        provider.get("missing", expected_capability_role="image_codec")
    with pytest.raises(ValueError, match=r"has capability role.*expected"):
        provider.get("codec", expected_capability_role="wrong_role")

    assert factory_calls == 0


def test_provider_requires_exact_descriptor_state_asset_names() -> None:
    with pytest.raises(ValueError, match=r"missing=.*training_codec"):
        InferenceAssetProvider(
            descriptors=_descriptors(),
            state_dicts={},
            device=torch.device("cpu"),
            model_factory=lambda component: InferenceAssetModule(
                **component.params
            ),
        )
    with pytest.raises(ValueError, match=r"unexpected=.*training_teacher"):
        InferenceAssetProvider(
            descriptors=_descriptors(),
            state_dicts={
                "training_codec": _state(),
                "training_teacher": _state(),
            },
            device=torch.device("cpu"),
            model_factory=lambda component: InferenceAssetModule(
                **component.params
            ),
        )


@pytest.mark.parametrize(
    ("state", "error_type", "message"),
    [
        ({}, ValueError, "keys do not match runtime"),
        ({"scale": torch.zeros(3)}, ValueError, "shape does not match runtime"),
        (
            {"scale": torch.zeros(2, dtype=torch.float64)},
            ValueError,
            "dtype does not match runtime",
        ),
        (
            {
                "scale": torch.sparse_coo_tensor(
                    torch.tensor([[0, 1]]),
                    torch.ones(2),
                    size=(2,),
                    check_invariants=True,
                )
            },
            ValueError,
            "layout does not match runtime",
        ),
        (
            cast(dict[str, torch.Tensor], {1: torch.zeros(2)}),
            TypeError,
            "keys must be exact strings",
        ),
    ],
)
def test_provider_validates_state_before_loading_and_does_not_cache_failure(
    state: Mapping[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    factory_calls = 0

    def model_factory(component: ComponentConfig) -> nn.Module:
        nonlocal factory_calls
        factory_calls += 1
        return InferenceAssetModule(**component.params)

    provider = InferenceAssetProvider(
        descriptors=_descriptors(),
        state_dicts={"training_codec": state},
        device=torch.device("cpu"),
        model_factory=model_factory,
    )

    for expected_calls in (1, 2):
        with pytest.raises(error_type, match=message):
            provider.get("codec", expected_capability_role="image_codec")
        assert factory_calls == expected_calls


def test_provider_rejects_non_module_factory_result_without_caching() -> None:
    factory_calls = 0

    def model_factory(component: ComponentConfig) -> nn.Module:
        nonlocal factory_calls
        del component
        factory_calls += 1
        return cast(Any, object())

    provider = InferenceAssetProvider(
        descriptors=_descriptors(),
        state_dicts={"training_codec": _state()},
        device=torch.device("cpu"),
        model_factory=model_factory,
    )

    for expected_calls in (1, 2):
        with pytest.raises(TypeError, match=r"factory must return nn\.Module"):
            provider.get("codec", expected_capability_role="image_codec")
        assert factory_calls == expected_calls


def test_provider_rejects_lazy_runtime_state_without_caching() -> None:
    factory_calls = 0

    def model_factory(component: ComponentConfig) -> nn.Module:
        nonlocal factory_calls
        del component
        factory_calls += 1
        return nn.LazyLinear(out_features=2)

    provider = InferenceAssetProvider(
        descriptors=_descriptors(component_name="lazy_linear"),
        state_dicts={
            "training_codec": {
                "weight": torch.zeros((3, 4)),
                "bias": torch.zeros(3),
            }
        },
        device=torch.device("cpu"),
        model_factory=model_factory,
    )

    for expected_calls in (1, 2):
        with pytest.raises(
            ValueError,
            match=r"runtime state is lazy.*exact shape validation",
        ):
            provider.get("codec", expected_capability_role="image_codec")
        assert factory_calls == expected_calls


def test_provider_propagates_unknown_registered_component_failure() -> None:
    provider = InferenceAssetProvider(
        descriptors=_descriptors(component_name="missing_model"),
        state_dicts={"training_codec": _state()},
        device=torch.device("cpu"),
        model_factory=build_model,
    )

    with pytest.raises(RegistryError, match="missing_model"):
        provider.get("codec", expected_capability_role="image_codec")


def test_empty_provider_supports_legacy_sampling_contexts() -> None:
    provider = InferenceAssetProvider.empty()

    with pytest.raises(KeyError, match="available: <none>"):
        provider.get("codec", expected_capability_role="image_codec")

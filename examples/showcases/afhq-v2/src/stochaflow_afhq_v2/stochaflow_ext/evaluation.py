"""Formal AFHQ-v2 class-aware generation evaluation components."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import torch
from torch import nn
from torchmetrics import Metric

from stochaflow.evaluation import (
    EvaluationBuilder,
    EvaluationPlan,
    EvaluationSamplingCapability,
    EvaluationSamplingRequest,
    EvaluationStepOutput,
    JsonlPredictionArtifactSink,
    PredictionRecord,
    PredictionSampleIdentity,
    ResolvedCheckpointSubject,
    ResolvedPredictionArtifactSubject,
)
from stochaflow.metrics import MetricSpec, MetricUpdate, build_metric
from stochaflow.sampling import SamplingOutput
from stochaflow.utils.registry import REGISTRIES

AFHQ_V2_EVALUATION_BUILDER = "afhq-v2.class-conditional-generation"
AFHQ_V2_DISTRIBUTION_METRIC = "afhq-v2.class-aware-distribution"
AFHQ_V2_IMAGE_PAIR_CHANNEL = "afhq-v2.image-pairs"
AFHQ_V2_IMAGE_CODEC = "stochaflow.afhq-v2.rgb-u8-base64.v1"
AFHQ_V2_CLASS_MAPPING = MappingProxyType({"cat": 0, "dog": 1, "wild": 2})
AFHQ_V2_IMAGE_SHAPE = (3, 128, 128)


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if type(value) is not dict and type(value) is not MappingProxyType:
        raise TypeError(f"{path} must be an exact mapping")
    result = dict(cast(Mapping[object, object], value))
    if any(type(key) is not str for key in result):
        raise TypeError(f"{path} keys must be exact strings")
    return cast(dict[str, Any], result)


def _positive_integer(value: object, *, path: str) -> int:
    if type(value) is not int or cast(int, value) <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return cast(int, value)


def _class_mapping(value: object, *, path: str) -> dict[str, int]:
    raw = _mapping(value, path=path)
    if raw != dict(AFHQ_V2_CLASS_MAPPING):
        raise ValueError(
            f"{path} must be the official AFHQ-v2 cat/dog/wild mapping"
        )
    return cast(dict[str, int], raw)


def _class_counts(
    value: object,
    *,
    class_mapping: Mapping[str, int],
    path: str,
) -> dict[str, int]:
    raw = _mapping(value, path=path)
    if set(raw) != set(class_mapping):
        raise ValueError(f"{path} keys must exactly match class_mapping")
    return {
        name: _positive_integer(raw[name], path=f"{path}.{name}")
        for name in class_mapping
    }


def _provider_specs(value: object) -> tuple[tuple[str, dict[str, Any]], ...]:
    if type(value) not in {list, tuple} or not value:
        raise ValueError("AFHQ-v2 metric providers must be a non-empty list")
    providers: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, declared in enumerate(cast(Sequence[object], value)):
        raw = _mapping(declared, path=f"AFHQ-v2 metric providers[{index}]")
        if set(raw) != {"name", "params"}:
            raise ValueError(
                "AFHQ-v2 metric provider must contain exactly name and params"
            )
        name = raw["name"]
        if type(name) is not str or not cast(str, name).strip():
            raise ValueError("AFHQ-v2 metric provider name must be non-empty")
        if name in seen:
            raise ValueError(f"duplicate AFHQ-v2 metric provider {name!r}")
        seen.add(cast(str, name))
        params = _mapping(
            raw["params"],
            path=f"AFHQ-v2 metric provider {name!r} params",
        )
        providers.append((cast(str, name), params))
    return tuple(providers)


@REGISTRIES.metrics.register(AFHQ_V2_DISTRIBUTION_METRIC)
class AFHQV2ClassAwareDistributionMetric(Metric):
    """Bind formal reference metrics to aggregate and per-class AFHQ scopes."""

    real_counts: torch.Tensor
    fake_counts: torch.Tensor

    def __init__(
        self,
        *,
        class_mapping: Mapping[str, int],
        expected_real: Mapping[str, int],
        expected_fake: Mapping[str, int],
        providers: list[dict[str, Any]],
    ) -> None:
        super().__init__(sync_on_compute=False)
        self.class_mapping = _class_mapping(
            class_mapping,
            path="AFHQ-v2 metric class_mapping",
        )
        self.expected_real = _class_counts(
            expected_real,
            class_mapping=self.class_mapping,
            path="AFHQ-v2 metric expected_real",
        )
        self.expected_fake = _class_counts(
            expected_fake,
            class_mapping=self.class_mapping,
            path="AFHQ-v2 metric expected_fake",
        )
        self.provider_specs = _provider_specs(providers)
        self._scope_names = ("aggregate", *self.class_mapping)
        self._metric_names: dict[tuple[str, str], str] = {}
        for scope_index, scope in enumerate(self._scope_names):
            for provider_index, (provider_name, params) in enumerate(
                self.provider_specs
            ):
                module_name = f"quality_{scope_index}_{provider_index}"
                metric = build_metric(
                    MetricSpec(
                        id=module_name,
                        name=provider_name,
                        channel=AFHQ_V2_IMAGE_PAIR_CHANNEL,
                        params=dict(params),
                    )
                )
                self.add_module(module_name, metric)
                self._metric_names[(scope, provider_name)] = module_name
        class_count = len(self.class_mapping)
        self.add_state(
            "real_counts",
            default=torch.zeros(class_count, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_counts",
            default=torch.zeros(class_count, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def _metric(self, scope: str, provider: str) -> Metric:
        value = cast(object, getattr(self, self._metric_names[(scope, provider)]))
        if not isinstance(value, Metric):
            raise TypeError("AFHQ-v2 nested quality provider must be a Metric")
        return value

    def update(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        real: bool,
    ) -> None:
        if images.ndim != 4:
            raise TypeError("AFHQ-v2 metric images must be a rank-4 Tensor")
        if tuple(images.shape[1:]) != AFHQ_V2_IMAGE_SHAPE:
            raise ValueError("AFHQ-v2 metric images must be RGB 128x128")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("AFHQ-v2 metric images must be finite floating values")
        if images.numel() and (
            float(images.min()) < 0.0 or float(images.max()) > 1.0
        ):
            raise ValueError("AFHQ-v2 metric images must lie in [0, 1]")
        if (
            labels.ndim != 1
            or labels.dtype != torch.long
            or labels.shape[0] != images.shape[0]
        ):
            raise TypeError("AFHQ-v2 metric labels must be matching 1D long values")
        if type(real) is not bool:
            raise TypeError("AFHQ-v2 metric real flag must be an exact bool")
        known_labels = set(self.class_mapping.values())
        observed_labels = set(labels.detach().cpu().tolist())
        if not observed_labels <= known_labels:
            raise ValueError("AFHQ-v2 metric received an unknown class label")
        images = images.to(self.real_counts.device)
        labels = labels.to(self.real_counts.device)
        counter = self.real_counts if real else self.fake_counts
        for name, label in self.class_mapping.items():
            mask = labels == label
            count = int(mask.sum())
            if count:
                counter[label] += count
                selected = images[mask]
                for provider_name, _ in self.provider_specs:
                    self._metric(name, provider_name).update(
                        selected,
                        real=real,
                    )
        for provider_name, _ in self.provider_specs:
            self._metric("aggregate", provider_name).update(images, real=real)

    def compute(self) -> Mapping[str, torch.Tensor | float]:
        self._validate_counts(self.real_counts, self.expected_real, kind="real")
        self._validate_counts(self.fake_counts, self.expected_fake, kind="fake")
        values: dict[str, torch.Tensor | float] = {}
        for scope in self._scope_names:
            for provider_name, _ in self.provider_specs:
                result = cast(object, self._metric(scope, provider_name).compute())
                if isinstance(result, Mapping):
                    if not result:
                        raise ValueError(
                            f"AFHQ-v2 provider {provider_name!r} returned no values"
                        )
                    for subkey, value in result.items():
                        if type(subkey) is not str or not cast(str, subkey):
                            raise TypeError(
                                "AFHQ-v2 provider result keys must be non-empty"
                            )
                        key = f"{scope}.{provider_name}_{subkey}"
                        values[key] = cast(torch.Tensor | float, value)
                else:
                    values[f"{scope}.{provider_name}"] = cast(
                        torch.Tensor | float,
                        result,
                    )
        return values

    def _validate_counts(
        self,
        observed: torch.Tensor,
        expected: Mapping[str, int],
        *,
        kind: str,
    ) -> None:
        actual = {
            name: int(observed[label].detach().cpu())
            for name, label in self.class_mapping.items()
        }
        if actual != dict(expected):
            raise ValueError(
                f"AFHQ-v2 {kind} class counts are incomplete: "
                f"expected={dict(expected)}, observed={actual}"
            )

    def reset(self) -> None:
        super().reset()
        for scope in self._scope_names:
            for provider_name, _ in self.provider_specs:
                self._metric(scope, provider_name).reset()


@dataclass(frozen=True, slots=True)
class AFHQV2EvaluationProfile:
    """Strict task-owned sampling and evidence identity for one AFHQ run."""

    class_mapping: Mapping[str, int]
    expected_per_class: Mapping[str, int]
    sampling: EvaluationSamplingRequest
    conditions: tuple[tuple[int, int], ...]
    gallery_count: int


def _evaluation_profile(
    value: Mapping[str, Any],
    *,
    expected_examples: int,
) -> AFHQV2EvaluationProfile:
    raw = _mapping(value, path="AFHQ-v2 evaluation params")
    allowed = {
        "class_mapping",
        "expected_per_class",
        "sampling",
        "gallery_count",
    }
    if set(raw) != allowed:
        raise ValueError(
            "AFHQ-v2 evaluation params must contain exactly "
            + ", ".join(sorted(allowed))
        )
    mapping = _class_mapping(
        raw["class_mapping"],
        path="AFHQ-v2 evaluation class_mapping",
    )
    counts = _class_counts(
        raw["expected_per_class"],
        class_mapping=mapping,
        path="AFHQ-v2 evaluation expected_per_class",
    )
    if sum(counts.values()) != expected_examples:
        raise ValueError(
            "AFHQ-v2 expected_per_class must sum to protocol.expected_examples"
        )
    sampling = _mapping(raw["sampling"], path="AFHQ-v2 evaluation sampling")
    if set(sampling) != {
        "recipe",
        "sampler",
        "options",
        "shape",
        "num_samples",
        "batch_size",
        "seed",
    }:
        raise ValueError("AFHQ-v2 sampling fields are incomplete or unknown")
    recipe = _mapping(
        sampling["recipe"],
        path="AFHQ-v2 sampling recipe",
    )
    if set(recipe) != {"name", "contract"}:
        raise ValueError("AFHQ-v2 sampling recipe requires name and contract")
    recipe_name = recipe["name"]
    if type(recipe_name) is not str or not cast(str, recipe_name).strip():
        raise ValueError("AFHQ-v2 sampling recipe name must be non-empty")
    recipe_contract = _mapping(
        recipe["contract"],
        path="AFHQ-v2 sampling recipe contract",
    )
    shape_value = sampling["shape"]
    if not isinstance(shape_value, (list, tuple)):
        raise TypeError("AFHQ-v2 sampling shape must be a sequence")
    shape = tuple(cast(Sequence[int], shape_value))
    if shape != AFHQ_V2_IMAGE_SHAPE:
        raise ValueError("AFHQ-v2 sampling shape must be [3, 128, 128]")
    options = _mapping(
        sampling["options"],
        path="AFHQ-v2 sampling options",
    )
    conditions_value = options.get("conditions")
    if type(conditions_value) not in {list, tuple} or not conditions_value:
        raise ValueError("AFHQ-v2 sampling options.conditions must be non-empty")
    conditions: list[tuple[int, int]] = []
    for index, declared in enumerate(
        cast(Sequence[object], conditions_value)
    ):
        condition = _mapping(
            declared,
            path=f"AFHQ-v2 sampling conditions[{index}]",
        )
        if set(condition) != {"class_label", "count"}:
            raise ValueError(
                "AFHQ-v2 sampling conditions require class_label and count"
            )
        label = condition["class_label"]
        if type(label) is not int:
            raise TypeError("AFHQ-v2 sampling class_label must be an exact integer")
        count = _positive_integer(
            condition["count"],
            path=f"AFHQ-v2 sampling conditions[{index}].count",
        )
        conditions.append((cast(int, label), count))
    if len({label for label, _ in conditions}) != len(mapping):
        raise ValueError("AFHQ-v2 sampling must declare every class exactly once")
    condition_counts = {
        next(name for name, value in mapping.items() if value == label): count
        for label, count in conditions
        if label in set(mapping.values())
    }
    if condition_counts != counts:
        raise ValueError(
            "AFHQ-v2 sampling condition counts must match expected_per_class"
        )
    request = EvaluationSamplingRequest(
        options=options,
        sampler=_mapping(
            sampling["sampler"],
            path="AFHQ-v2 sampling sampler",
        ),
        shape=shape,
        num_samples=_positive_integer(
            sampling["num_samples"],
            path="AFHQ-v2 sampling num_samples",
        ),
        batch_size=_positive_integer(
            sampling["batch_size"],
            path="AFHQ-v2 sampling batch_size",
        ),
        seed=cast(int, sampling["seed"]),
        expected_recipe_name=cast(str, recipe_name),
        expected_recipe_contract=recipe_contract,
    )
    if request.num_samples != expected_examples:
        raise ValueError(
            "AFHQ-v2 sampling num_samples must equal protocol.expected_examples"
        )
    gallery_count = _positive_integer(
        raw["gallery_count"],
        path="AFHQ-v2 evaluation gallery_count",
    )
    if gallery_count > expected_examples:
        raise ValueError("AFHQ-v2 gallery_count exceeds the evaluation sample plan")
    return AFHQV2EvaluationProfile(
        class_mapping=MappingProxyType(mapping),
        expected_per_class=MappingProxyType(counts),
        sampling=request,
        conditions=tuple(conditions),
        gallery_count=gallery_count,
    )


def _validate_offline_generation_profile(
    subject: ResolvedPredictionArtifactSubject,
    current: AFHQV2EvaluationProfile,
    *,
    expected_examples: int,
) -> None:
    """Bind offline scoring to the generation request that produced its bytes."""

    inference_profile = _mapping(
        subject.inputs.inference_profile,
        path="AFHQ-v2 prediction artifact inference_profile",
    )
    if "evaluation" not in inference_profile:
        raise ValueError(
            "AFHQ-v2 prediction artifact inference_profile is missing evaluation"
        )
    evaluation = _mapping(
        inference_profile["evaluation"],
        path="AFHQ-v2 prediction artifact inference_profile.evaluation",
    )
    if set(evaluation) != {"name", "params"}:
        raise ValueError(
            "AFHQ-v2 prediction artifact evaluation identity requires name and params"
        )
    if evaluation["name"] != AFHQ_V2_EVALUATION_BUILDER:
        raise ValueError(
            "AFHQ-v2 prediction artifact was produced by a different evaluation "
            "builder"
        )
    producer = _evaluation_profile(
        _mapping(
            evaluation["params"],
            path="AFHQ-v2 prediction artifact evaluation params",
        ),
        expected_examples=expected_examples,
    )
    if producer.sampling != current.sampling:
        raise ValueError(
            "AFHQ-v2 offline generation profile does not match the prediction "
            "artifact"
        )


def _sample_plan(
    protocol_id: str,
    count: int,
    *,
    split: str,
) -> tuple[PredictionSampleIdentity, ...]:
    if split not in {"validation", "test"}:
        raise ValueError("AFHQ-v2 evaluation split must be validation or test")
    input_split = "official-test" if split == "test" else "validation"
    return tuple(
        PredictionSampleIdentity(
            sample_id=f"{protocol_id}:generated:{index:06d}",
            input_id=f"afhq-v2:{input_split}:{index:06d}",
            replicate_index=0,
        )
        for index in range(count)
    )


def _image_bytes(image: torch.Tensor) -> bytes:
    if image.ndim != 3:
        raise TypeError("AFHQ-v2 artifact image must be a rank-3 Tensor")
    if tuple(image.shape) != AFHQ_V2_IMAGE_SHAPE:
        raise ValueError("AFHQ-v2 artifact image must be RGB 128x128")
    image = image.detach().float().cpu()
    if not torch.isfinite(image).all():
        raise ValueError("AFHQ-v2 artifact image must be finite")
    quantized = ((image.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return quantized.contiguous().numpy().tobytes()


def _encoded_image(image: torch.Tensor) -> dict[str, Any]:
    payload = _image_bytes(image)
    return {
        "codec": AFHQ_V2_IMAGE_CODEC,
        "shape": list(AFHQ_V2_IMAGE_SHAPE),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _decoded_image(value: object, *, path: str) -> torch.Tensor:
    raw = _mapping(value, path=path)
    if set(raw) != {"codec", "shape", "sha256", "data"}:
        raise ValueError(f"{path} fields are incomplete or unknown")
    if raw["codec"] != AFHQ_V2_IMAGE_CODEC:
        raise ValueError(f"{path}.codec is unsupported")
    if tuple(cast(Sequence[object], raw["shape"])) != AFHQ_V2_IMAGE_SHAPE:
        raise ValueError(f"{path}.shape must be [3, 128, 128]")
    encoded = raw["data"]
    expected_bytes = math.prod(AFHQ_V2_IMAGE_SHAPE)
    expected_encoded = 4 * math.ceil(expected_bytes / 3)
    if type(encoded) is not str or len(cast(str, encoded)) != expected_encoded:
        raise ValueError(f"{path}.data has an invalid encoded size")
    try:
        payload = base64.b64decode(cast(str, encoded), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{path}.data is not canonical base64") from exc
    if len(payload) != expected_bytes:
        raise ValueError(f"{path}.data has an invalid decoded size")
    digest = raw["sha256"]
    if type(digest) is not str or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{path}.sha256 does not match image bytes")
    values = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    return values.reshape(AFHQ_V2_IMAGE_SHAPE).float().div_(255.0)


def _pair_record(
    identity: PredictionSampleIdentity,
    *,
    real: torch.Tensor,
    fake: torch.Tensor,
    class_name: str,
    class_label: int,
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=identity.sample_id,
        input_id=identity.input_id,
        replicate_index=identity.replicate_index,
        payload={
            "class_name": class_name,
            "class_label": class_label,
            "reference": _encoded_image(real),
            "prediction": _encoded_image(fake),
        },
    )


def _record_pair(
    record: PredictionRecord,
    *,
    class_mapping: Mapping[str, int],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    payload = record.payload
    class_name = payload.get("class_name")
    class_label = payload.get("class_label")
    if type(class_name) is not str or class_name not in class_mapping:
        raise ValueError("AFHQ-v2 prediction record class_name is invalid")
    if type(class_label) is not int or class_mapping[class_name] != class_label:
        raise ValueError("AFHQ-v2 prediction record class identity is inconsistent")
    real = _decoded_image(
        payload.get("reference"),
        path="AFHQ-v2 prediction record reference",
    )
    fake = _decoded_image(
        payload.get("prediction"),
        path="AFHQ-v2 prediction record prediction",
    )
    return real, fake, cast(int, class_label)


class AFHQV2GenerationEvaluator:
    """Generate once for live evaluation or decode the same records offline."""

    metric_channels = frozenset({AFHQ_V2_IMAGE_PAIR_CHANNEL})

    def __init__(
        self,
        *,
        profile: AFHQV2EvaluationProfile,
        sample_plan: tuple[PredictionSampleIdentity, ...],
        sampling: EvaluationSamplingCapability | None,
    ) -> None:
        self.profile = profile
        self.sample_plan = sample_plan
        self.sampling = sampling
        self._generated: dict[int, torch.Tensor] | None = None
        self._class_offsets = dict.fromkeys(profile.class_mapping.values(), 0)
        self._position = 0

    def _generate(self) -> dict[int, torch.Tensor]:
        if self.sampling is None:
            raise RuntimeError("offline AFHQ-v2 evaluator cannot invoke sampling")
        output = cast(object, self.sampling.execute(self.profile.sampling))
        if not isinstance(output, SamplingOutput):
            raise TypeError("AFHQ-v2 sampling capability returned an invalid output")
        parts: list[torch.Tensor] = []
        for index, batch in enumerate(output.batches):
            samples = cast(object, batch.samples)
            if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
                raise TypeError(
                    f"AFHQ-v2 sampling batch {index} must contain NCHW Tensor samples"
                )
            if tuple(samples.shape[1:]) != AFHQ_V2_IMAGE_SHAPE:
                raise ValueError("AFHQ-v2 generated samples must be RGB 128x128")
            if not torch.isfinite(samples).all():
                raise ValueError("AFHQ-v2 generated samples must be finite")
            parts.append(samples.detach().cpu())
        generated = torch.cat(parts, dim=0)
        if generated.shape[0] != self.profile.sampling.num_samples:
            raise ValueError("AFHQ-v2 sampling output count is incomplete")
        split: dict[int, torch.Tensor] = {}
        offset = 0
        for label, count in self.profile.conditions:
            split[label] = generated[offset : offset + count]
            offset += count
        return split

    def _live_records(self, batch: object) -> tuple[PredictionRecord, ...]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise TypeError(
                "AFHQ-v2 live batches must be (images, {'class_label': labels})"
            )
        images = cast(object, batch[0])
        conditions = cast(object, batch[1])
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise TypeError("AFHQ-v2 live images must be a rank-4 Tensor")
        if not isinstance(conditions, Mapping) or set(conditions) != {"class_label"}:
            raise TypeError("AFHQ-v2 live conditions must contain only class_label")
        labels = cast(object, conditions["class_label"])
        if (
            not isinstance(labels, torch.Tensor)
            or labels.ndim != 1
            or labels.dtype != torch.long
            or labels.shape[0] != images.shape[0]
        ):
            raise TypeError("AFHQ-v2 live labels must be matching 1D long values")
        if self._generated is None:
            self._generated = self._generate()
        label_to_name = {
            label: name for name, label in self.profile.class_mapping.items()
        }
        records: list[PredictionRecord] = []
        for image, label_tensor in zip(images, labels, strict=True):
            if self._position >= len(self.sample_plan):
                raise ValueError("AFHQ-v2 live data exceeds the declared sample plan")
            label = int(label_tensor)
            if label not in label_to_name:
                raise ValueError("AFHQ-v2 live data contains an unknown class label")
            offset = self._class_offsets[label]
            fake_group = self._generated[label]
            if offset >= fake_group.shape[0]:
                raise ValueError(
                    f"AFHQ-v2 live data exceeds generated class allocation {label}"
                )
            records.append(
                _pair_record(
                    self.sample_plan[self._position],
                    real=image,
                    fake=fake_group[offset],
                    class_name=label_to_name[label],
                    class_label=label,
                )
            )
            self._class_offsets[label] = offset + 1
            self._position += 1
        return tuple(records)

    def evaluate_batch(self, batch: Any) -> EvaluationStepOutput:
        live = not isinstance(batch, PredictionRecord)
        records = self._live_records(batch) if live else (batch,)
        expected = self.sample_plan[
            self._position - len(records) if live else self._position
            : self._position if live else self._position + len(records)
        ]
        if not live:
            self._position += len(records)
        if tuple(record.identity for record in records) != expected:
            raise ValueError(
                "AFHQ-v2 prediction record identities must match the sample plan"
            )
        decoded = tuple(
            _record_pair(record, class_mapping=self.profile.class_mapping)
            for record in records
        )
        real = torch.stack([item[0] for item in decoded])
        fake = torch.stack([item[1] for item in decoded])
        labels = torch.tensor([item[2] for item in decoded], dtype=torch.long)
        return EvaluationStepOutput(
            num_examples=len(records),
            sample_ids=tuple(record.sample_id for record in records),
            metric_update_groups=(
                {
                    AFHQ_V2_IMAGE_PAIR_CHANNEL: MetricUpdate(
                        args=(real, labels),
                        kwargs={"real": True},
                    )
                },
                {
                    AFHQ_V2_IMAGE_PAIR_CHANNEL: MetricUpdate(
                        args=(fake, labels),
                        kwargs={"real": False},
                    )
                },
            ),
            records=records if live else None,
        )


@REGISTRIES.evaluation_builders.register(AFHQ_V2_EVALUATION_BUILDER)
class AFHQV2GenerationEvaluationBuilder(EvaluationBuilder):
    """Compose one split-specific AFHQ-v2 generation quality profile."""

    def build(self) -> EvaluationPlan:
        profile = _evaluation_profile(
            self.context.params,
            expected_examples=self.context.protocol.expected_examples,
        )
        self._validate_metric_bindings(profile)
        split = self.context.data_identity.get("split")
        if split not in {"validation", "test"}:
            raise ValueError(
                "AFHQ-v2 evaluation data identity must declare validation or test"
            )
        sample_plan = _sample_plan(
            self.context.protocol.id,
            self.context.protocol.expected_examples,
            split=cast(str, split),
        )
        sink = None
        modules: Mapping[str, nn.Module]
        sampling: EvaluationSamplingCapability | None
        if isinstance(self.context.subject, ResolvedCheckpointSubject):
            if not isinstance(cast(object, self.context.inference), nn.Module):
                raise TypeError("AFHQ-v2 live evaluation requires a primary model")
            if self.context.sampling is None:
                raise TypeError(
                    "AFHQ-v2 live evaluation requires checkpoint sampling capability"
                )
            if self.context.artifact_root is None:
                raise ValueError("AFHQ-v2 live evaluation requires artifact staging")
            sampling = self.context.sampling
            modules = {"primary": cast(nn.Module, self.context.inference)}
            manifest_split = (
                "official-test" if split == "test" else "validation"
            )
            sink = JsonlPredictionArtifactSink(
                self.context.artifact_root,
                expected_samples=sample_plan,
                preprocess={
                    "reference": (
                        "authenticated AFHQ-v2 "
                        f"{manifest_split} manifest order"
                    ),
                    "source_range": [-1.0, 1.0],
                    "metric_range": [0.0, 1.0],
                    "pairing": f"same-class {manifest_split} allocation",
                },
                postprocess={
                    "codec": AFHQ_V2_IMAGE_CODEC,
                    "quantization": "clamp[-1,1]-round-rgb-u8",
                },
                gallery_count=profile.gallery_count,
            )
        elif isinstance(
            self.context.subject,
            ResolvedPredictionArtifactSubject,
        ):
            if self.context.inference is not None or self.context.sampling is not None:
                raise ValueError(
                    "AFHQ-v2 offline evaluation must not receive live inference"
                )
            _validate_offline_generation_profile(
                self.context.subject,
                profile,
                expected_examples=self.context.protocol.expected_examples,
            )
            sampling = None
            modules = {}
        else:
            raise TypeError("AFHQ-v2 evaluation subject is unsupported")
        evaluator = AFHQV2GenerationEvaluator(
            profile=profile,
            sample_plan=sample_plan,
            sampling=sampling,
        )
        return EvaluationPlan(
            evaluator=evaluator,
            data=self.context.data,
            metric_specs=self.context.metric_specs,
            protocol=self.context.protocol,
            subject=self.context.subject,
            data_identity=self.context.data_identity,
            artifact_sink=sink,
            modules=modules,
        )

    def _validate_metric_bindings(
        self,
        profile: AFHQV2EvaluationProfile,
    ) -> None:
        if len(self.context.metric_specs) != 1:
            raise ValueError("AFHQ-v2 profile requires exactly one distribution metric")
        spec = self.context.metric_specs[0]
        if spec.name != AFHQ_V2_DISTRIBUTION_METRIC:
            raise ValueError("AFHQ-v2 profile requires its class-aware metric")
        if spec.channel != AFHQ_V2_IMAGE_PAIR_CHANNEL:
            raise ValueError("AFHQ-v2 metric must bind the image-pair channel")
        params = spec.params
        if set(params) != {
            "class_mapping",
            "expected_real",
            "expected_fake",
            "providers",
        }:
            raise ValueError("AFHQ-v2 metric params are incomplete or unknown")
        metric_mapping = _class_mapping(
            params.get("class_mapping"),
            path="AFHQ-v2 metric class_mapping",
        )
        expected_real = _class_counts(
            params.get("expected_real"),
            class_mapping=metric_mapping,
            path="AFHQ-v2 metric expected_real",
        )
        expected_fake = _class_counts(
            params.get("expected_fake"),
            class_mapping=metric_mapping,
            path="AFHQ-v2 metric expected_fake",
        )
        _provider_specs(params.get("providers"))
        if metric_mapping != dict(profile.class_mapping):
            raise ValueError("AFHQ-v2 evaluator and metric class mappings differ")
        if expected_real != dict(profile.expected_per_class):
            raise ValueError("AFHQ-v2 evaluator and metric real allocations differ")
        if expected_fake != dict(profile.expected_per_class):
            raise ValueError("AFHQ-v2 evaluator and metric fake allocations differ")


__all__ = [
    "AFHQ_V2_CLASS_MAPPING",
    "AFHQ_V2_DISTRIBUTION_METRIC",
    "AFHQ_V2_EVALUATION_BUILDER",
    "AFHQ_V2_IMAGE_CODEC",
    "AFHQ_V2_IMAGE_PAIR_CHANNEL",
    "AFHQV2ClassAwareDistributionMetric",
    "AFHQV2GenerationEvaluationBuilder",
]

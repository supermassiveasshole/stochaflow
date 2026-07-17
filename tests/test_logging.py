"""Tests for experiment logger media fan-out."""

from pathlib import Path
from typing import Any

from PIL import Image

from stochaflow.utils import logging as logging_module
from stochaflow.utils.logging import (
    CompositeLogger,
    ExperimentLogger,
    LocalLogger,
    TensorBoardLogger,
    WandbLogger,
)


class RecordingBackend(ExperimentLogger):
    def __init__(self) -> None:
        self.images: list[tuple[str, Path, int, str | None]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del metrics, step

    def log_image(
        self,
        tag: str,
        path: str | Path,
        *,
        step: int,
        caption: str | None = None,
    ) -> None:
        self.images.append((tag, Path(path), step, caption))

    def close(self) -> None:
        return None


def _image(path: Path) -> Path:
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path)
    return path


def test_composite_logger_fans_out_images(tmp_path) -> None:
    first = RecordingBackend()
    second = RecordingBackend()
    logger = CompositeLogger([first, second])
    path = _image(tmp_path / "sample.png")

    logger.log_image("diagnostics/sample", path, step=7, caption="preview")

    assert first.images == second.images
    assert first.images == [("diagnostics/sample", path, 7, "preview")]


def test_local_logger_records_image_path_and_caption(tmp_path) -> None:
    path = _image(tmp_path / "sample.png")
    logger = LocalLogger(
        output_dir=str(tmp_path),
        run_name="test",
        console=False,
    )

    logger.log_image("diagnostics/sample", path, step=3, caption="preview")
    logger.close()

    text = (tmp_path / "train.log").read_text(encoding="utf-8")
    assert str(path) in text
    assert "caption=preview" in text


def test_tensorboard_logger_writes_image_summary(tmp_path) -> None:
    path = _image(tmp_path / "sample.png")
    logger = TensorBoardLogger(output_dir=str(tmp_path), run_name="test")

    logger.log_image("diagnostics/sample", path, step=5, caption="preview")
    logger.close()

    event_files = list((tmp_path / "tensorboard" / "test").glob("events.out.tfevents.*"))
    assert event_files


def test_wandb_logger_uploads_image(monkeypatch, tmp_path) -> None:
    calls: list[tuple[dict[str, Any], int | None]] = []

    class FakeConfig:
        def update(self, config, allow_val_change=False) -> None:
            del config, allow_val_change

    class FakeRun:
        config = FakeConfig()

        def finish(self) -> None:
            return None

    class FakeWandb:
        def init(self, **kwargs):
            del kwargs
            return FakeRun()

        def Image(self, path: str, caption: str | None = None):
            return (path, caption)

        def log(self, payload: dict[str, Any], step: int | None = None) -> None:
            calls.append((payload, step))

    fake = FakeWandb()
    monkeypatch.setattr(
        logging_module.importlib,
        "import_module",
        lambda name: fake if name == "wandb" else None,
    )
    path = _image(tmp_path / "sample.png")
    logger = WandbLogger(output_dir=str(tmp_path), run_name="test")

    logger.log_image("diagnostics/sample", path, step=9, caption="preview")

    assert calls == [
        ({"diagnostics/sample": (str(path), "preview")}, 9),
    ]

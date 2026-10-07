"""MetricsCsvWriter: logged metrics must reach a plain CSV on disk.

Background: on a compute node the repo's Comet logger is forced offline and writes an
opaque archive, so a 4-day beta sweep produced no readable metrics — the trend could only
be recovered afterwards by loading checkpoint tensors. Lightning's own CSVLogger cannot be
used here: the repo CLI hardcodes `trainer.logger.init_args.offline_directory`, so passing
a *list* of loggers makes jsonargparse replace the list with a bare Namespace and the run
dies at instantiation (that is exactly how the Polaris smoke test failed). Hence a callback.

Discriminating properties, each with the wrong implementation it catches:
- writes/appends: a second epoch adds a row rather than a second header — catches
  reopening the file in "w" mode, which would leave only the last epoch.
- resume: a fresh writer over an existing file reuses that file's header and appends —
  catches truncating on restart, which is the *normal* case on a preemptable queue.
- sanity: sanity-check epochs are skipped — catches a leading row of untrained values.
- robust: non-scalar metrics are dropped rather than raising — catches float() blowing up
  mid-run and killing a multi-day job.
"""

import csv
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from hepattn.keras.callbacks import MetricsCsvWriter


def _trainer(epoch, step, metrics, sanity=False):
    return SimpleNamespace(current_epoch=epoch, global_step=step, callback_metrics=metrics, sanity_checking=sanity)


def _rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _header(path):
    with path.open() as f:
        return next(csv.reader(f))


def test_skips_sanity_check(tmp_path):
    path = tmp_path / "metrics.csv"
    MetricsCsvWriter(str(path)).on_validation_epoch_end(_trainer(0, 0, {"val/loss": torch.tensor(1.0)}, sanity=True), None)  # ty: ignore [invalid-argument-type]
    assert not path.exists(), "a sanity-check row would record untrained values as epoch 0"


def test_writes_then_appends(tmp_path):
    path = tmp_path / "metrics.csv"
    w = MetricsCsvWriter(str(path))
    w.on_validation_epoch_end(_trainer(0, 100, {"val/loss": torch.tensor(29.5), "val/bits_mean": torch.tensor(8.0)}), None)  # ty: ignore [invalid-argument-type]
    w.on_validation_epoch_end(_trainer(1, 200, {"val/loss": torch.tensor(30.1), "val/bits_mean": torch.tensor(7.98)}), None)  # ty: ignore [invalid-argument-type]

    rows = _rows(path)
    assert len(rows) == 2, f"expected 2 data rows, got {len(rows)}"
    assert float(rows[0]["val/bits_mean"]) == pytest.approx(8.0, abs=1e-5)
    assert float(rows[1]["val/bits_mean"]) == pytest.approx(7.98, abs=1e-5)
    assert float(rows[1]["epoch"]) == 1.0
    assert float(rows[1]["step"]) == 200.0


def test_resume_keeps_schema_and_appends(tmp_path):
    """A preempted job restarts with a fresh callback instance over an existing file."""
    path = tmp_path / "metrics.csv"
    MetricsCsvWriter(str(path)).on_validation_epoch_end(_trainer(0, 100, {"val/loss": torch.tensor(29.5)}), None)  # ty: ignore [invalid-argument-type]
    before = _header(path)

    resumed = MetricsCsvWriter(str(path))
    resumed.on_validation_epoch_end(_trainer(1, 200, {"val/loss": torch.tensor(30.0), "brand/new/key": torch.tensor(1.0)}), None)  # ty: ignore [invalid-argument-type]

    assert _header(path) == before, "resume must not rewrite the header"
    assert len(_rows(path)) == 2, "resume must append, not truncate"


def test_non_scalar_metric_does_not_raise(tmp_path):
    path = tmp_path / "metrics.csv"
    w = MetricsCsvWriter(str(path))
    w.on_validation_epoch_end(_trainer(0, 100, {"val/loss": torch.tensor(32.0), "conf/matrix": torch.zeros(3, 3)}), None)  # ty: ignore [invalid-argument-type]
    rows = _rows(path)
    assert len(rows) == 1
    assert float(rows[0]["val/loss"]) == pytest.approx(32.0, abs=1e-5)
    assert "conf/matrix" not in rows[0]

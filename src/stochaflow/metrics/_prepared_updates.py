"""Package-internal bridge for deferred metric update commits."""

from stochaflow.metrics.contracts import PreparedMetricUpdates
from stochaflow.metrics.runtime import MetricEngine


def commit_prepared_metric_updates(
    engine: MetricEngine,
    updates: PreparedMetricUpdates,
) -> None:
    """Commit detached payloads without widening the public metric API."""

    engine._update_prepared(updates)

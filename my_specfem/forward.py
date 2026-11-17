import numpy as np
from numpy.typing import NDArray

from .acquisition import DISPLACEMENT_CHANNELS, PointForce, Receiver, Seismogram
from .elastic import ElasticOperator
from .time_integration import initialize, newmark_step


def run_elastic(
    operator: ElasticOperator,
    source: PointForce,
    receivers: tuple[Receiver, ...],
    *,
    t0: float,
    dt: float,
    steps: int,
) -> Seismogram:
    if not np.isfinite(t0) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("t0 must be finite and dt must be finite and positive")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    station_ids = tuple(receiver.station_id for receiver in receivers)
    if not station_ids or len(set(station_ids)) != len(station_ids):
        raise ValueError("at least one receiver is required and station IDs must be unique")

    shape = (operator.grid.node_count, 3)
    force = np.zeros(shape)

    def acceleration(displacement: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        force.fill(0.0)
        source.add_to(force, time)
        return operator.acceleration(displacement, force)

    times = t0 + dt * np.arange(steps + 1)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("sample times must be finite and strictly increasing")
    values = np.empty((len(times), len(receivers), len(DISPLACEMENT_CHANNELS)))
    state = initialize(times[0], np.zeros(shape), np.zeros(shape), acceleration)
    for index, time in enumerate(times):
        if index > 0:
            state = newmark_step(state, time, acceleration)
        for station, receiver in enumerate(receivers):
            values[index, station] = receiver.stencil.sample(state.q)
    return Seismogram(times, station_ids, DISPLACEMENT_CHANNELS, values)

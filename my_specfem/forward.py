from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .acoustic import AcousticOperator
from .acquisition import (
    DISPLACEMENT_CHANNELS,
    PRESSURE_CHANNELS,
    AcousticPointSource,
    Channel,
    PointForce,
    Receiver,
    Seismogram,
)
from .elastic import ElasticOperator
from .time_integration import Acceleration, State, initialize, newmark_step


def run_elastic(
    operator: ElasticOperator,
    source: PointForce,
    receivers: tuple[Receiver, ...],
    *,
    t0: float,
    dt: float,
    steps: int,
) -> Seismogram:
    shape = (operator.grid.node_count, 3)
    force = np.zeros(shape)

    def acceleration(displacement: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        force.fill(0.0)
        source.add_to(force, time)
        return operator.acceleration(displacement, force)

    return _march_and_sample(
        np.zeros(shape), acceleration, lambda state: state.q, receivers,
        DISPLACEMENT_CHANNELS, t0=t0, dt=dt, steps=steps,
    )


def run_acoustic(
    operator: AcousticOperator,
    source: AcousticPointSource | None,
    receivers: tuple[Receiver, ...],
    *,
    t0: float,
    dt: float,
    steps: int,
    initial_potential: NDArray[np.float64] | None = None,
) -> Seismogram:
    shape = (operator.grid.node_count, 1)
    potential = (
        np.zeros(shape) if initial_potential is None else np.asarray(initial_potential, dtype=float)
    )
    if potential.shape != shape or not np.isfinite(potential).all():
        raise ValueError("initial potential must be finite and shaped (node_count, 1)")
    load = np.zeros(shape)

    def acceleration(potential: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        if source is not None:
            load.fill(0.0)
            source.add_to(load, time)
        return operator.acceleration(potential, load)

    return _march_and_sample(
        potential, acceleration, lambda state: -state.qddot, receivers,
        PRESSURE_CHANNELS, t0=t0, dt=dt, steps=steps,
    )


def _march_and_sample(
    initial_q: NDArray[np.float64],
    acceleration: Acceleration,
    observe: Callable[[State], NDArray[np.float64]],
    receivers: tuple[Receiver, ...],
    channels: tuple[Channel, ...],
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

    times = t0 + dt * np.arange(steps + 1)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("sample times must be finite and strictly increasing")
    values = np.empty((len(times), len(receivers), len(channels)))
    state = initialize(times[0], initial_q, np.zeros_like(initial_q), acceleration)
    for index, time in enumerate(times):
        if index > 0:
            state = newmark_step(state, time, acceleration)
        observable = observe(state)
        for station, receiver in enumerate(receivers):
            values[index, station] = receiver.stencil.sample(observable)
    return Seismogram(times, station_ids, channels, values)

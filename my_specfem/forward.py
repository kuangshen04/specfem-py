from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .acoustic import AcousticOperator, pressure_from_potential_acceleration
from .acquisition import (
    DISPLACEMENT_CHANNELS,
    PRESSURE_CHANNELS,
    AcousticPointSource,
    Channel,
    PointForce,
    Receiver,
    Seismogram,
)
from .coupling import FluidSolidInterface
from .elastic import ElasticOperator
from .time_integration import Acceleration, State, initialize, newmark_step


@dataclass(frozen=True)
class _Observation:
    receivers: tuple[Receiver, ...]
    channels: tuple[Channel, ...]
    field: Callable[[State], NDArray[np.float64]]


def run_elastic(
    operator: ElasticOperator,
    source: PointForce,
    receivers: tuple[Receiver, ...],
    *,
    t0: float,
    dt: float,
    steps: int,
) -> Seismogram:
    shape = (operator.domain.node_count, 3)
    force = np.zeros(shape)

    def acceleration(displacement: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        force.fill(0.0)
        source.add_to(force, time)
        return operator.acceleration(displacement, force)

    (seismogram,) = _march_and_sample(
        np.zeros(shape),
        acceleration,
        (_Observation(receivers, DISPLACEMENT_CHANNELS, lambda state: state.q),),
        t0=t0, dt=dt, steps=steps,
    )
    return seismogram


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
    shape = (operator.domain.node_count, 1)
    potential = _initial_scalar_field(initial_potential, shape)
    load = np.zeros(shape)

    def acceleration(potential: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        if source is not None:
            load.fill(0.0)
            source.add_to(load, time)
        return operator.acceleration(potential, load)

    (seismogram,) = _march_and_sample(
        potential,
        acceleration,
        (
            _Observation(
                receivers, PRESSURE_CHANNELS,
                lambda state: pressure_from_potential_acceleration(state.qddot),
            ),
        ),
        t0=t0, dt=dt, steps=steps,
    )
    return seismogram


def run_coupled(
    acoustic: AcousticOperator,
    elastic: ElasticOperator,
    interface: FluidSolidInterface,
    fluid_receivers: tuple[Receiver, ...],
    solid_receivers: tuple[Receiver, ...],
    *,
    t0: float,
    dt: float,
    steps: int,
    initial_potential: NDArray[np.float64] | None = None,
    initial_potential_rate: NDArray[np.float64] | None = None,
    fluid_source: AcousticPointSource | None = None,
    solid_source: PointForce | None = None,
) -> tuple[Seismogram, Seismogram]:
    """Advance both fields and return fluid pressure followed by solid displacement."""
    fluid_size = acoustic.domain.node_count
    fluid_shape = (fluid_size, 1)
    solid_shape = (elastic.domain.node_count, 3)
    potential = _initial_scalar_field(initial_potential, fluid_shape)
    potential_rate = _initial_scalar_field(initial_potential_rate, fluid_shape)
    solid_zero = np.zeros(np.prod(solid_shape))
    fluid_load = np.zeros(fluid_shape)
    solid_load = np.zeros(solid_shape)

    def acceleration(q: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        chi = q[:fluid_size].reshape(fluid_shape)
        displacement = q[fluid_size:].reshape(solid_shape)
        fluid_load.fill(0.0)
        solid_load.fill(0.0)
        if fluid_source is not None:
            fluid_source.add_to(fluid_load, time)
        if solid_source is not None:
            solid_source.add_to(solid_load, time)
        interface.add_fluid_load(fluid_load, displacement)
        chi_ddot = acoustic.acceleration(chi, fluid_load)
        interface.add_solid_load(solid_load, pressure_from_potential_acceleration(chi_ddot))
        u_ddot = elastic.acceleration(displacement, solid_load)
        return np.concatenate((chi_ddot.ravel(), u_ddot.ravel()))

    pressure, displacement = _march_and_sample(
        np.concatenate((potential.ravel(), solid_zero)),
        acceleration,
        (
            _Observation(
                fluid_receivers, PRESSURE_CHANNELS,
                lambda state: pressure_from_potential_acceleration(
                    state.qddot[:fluid_size].reshape(fluid_shape)
                ),
            ),
            _Observation(
                solid_receivers, DISPLACEMENT_CHANNELS,
                lambda state: state.q[fluid_size:].reshape(solid_shape),
            ),
        ),
        t0=t0, dt=dt, steps=steps,
        initial_qdot=np.concatenate((potential_rate.ravel(), solid_zero)),
    )
    return pressure, displacement


def _initial_scalar_field(
    values: NDArray[np.float64] | None, shape: tuple[int, int]
) -> NDArray[np.float64]:
    field = np.zeros(shape) if values is None else np.asarray(values, dtype=float)
    if field.shape != shape or not np.isfinite(field).all():
        raise ValueError("initial acoustic fields must be finite and shaped (node_count, 1)")
    return field


def _march_and_sample(
    initial_q: NDArray[np.float64],
    acceleration: Acceleration,
    observations: tuple[_Observation, ...],
    *,
    t0: float,
    dt: float,
    steps: int,
    initial_qdot: NDArray[np.float64] | None = None,
) -> tuple[Seismogram, ...]:
    if not np.isfinite(t0) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("t0 must be finite and dt must be finite and positive")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    times = t0 + dt * np.arange(steps + 1)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("sample times must be finite and strictly increasing")
    seismograms = []
    for observation in observations:
        station_ids = tuple(receiver.station_id for receiver in observation.receivers)
        if not station_ids or len(set(station_ids)) != len(station_ids):
            raise ValueError("each observation requires receivers with unique station IDs")
        values = np.empty((len(times), len(station_ids), len(observation.channels)))
        seismograms.append(Seismogram(times, station_ids, observation.channels, values))

    qdot = np.zeros_like(initial_q) if initial_qdot is None else initial_qdot
    state = initialize(times[0], initial_q, qdot, acceleration)
    for index, time in enumerate(times):
        if index > 0:
            state = newmark_step(state, time, acceleration)
        for observation, seismogram in zip(observations, seismograms):
            observable = observation.field(state)
            for station, receiver in enumerate(observation.receivers):
                seismogram.values[index, station] = receiver.stencil.sample(observable)
    return tuple(seismograms)

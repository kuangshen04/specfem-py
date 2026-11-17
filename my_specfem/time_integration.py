from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Acceleration = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


@dataclass(frozen=True)
class State:
    time: float
    q: NDArray[np.float64]
    qdot: NDArray[np.float64]
    qddot: NDArray[np.float64]


def initialize(
    time: float,
    q: NDArray[np.float64],
    qdot: NDArray[np.float64],
    acceleration: Acceleration,
) -> State:
    return State(time, q, qdot, acceleration(q, time))


def newmark_step(state: State, next_time: float, acceleration: Acceleration) -> State:
    """Explicit Newmark with beta=0 and gamma=1/2 for acceleration a(q, t)."""
    dt = next_time - state.time
    half_velocity = state.qdot + (0.5 * dt) * state.qddot
    q = state.q + dt * half_velocity
    qddot = acceleration(q, next_time)
    qdot = half_velocity + (0.5 * dt) * qddot
    return State(next_time, q, qdot, qddot)

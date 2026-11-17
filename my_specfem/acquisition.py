from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .sem import PointStencil


@dataclass(frozen=True)
class Ricker:
    frequency_hz: float
    peak_time_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("Ricker frequency must be finite and positive")
        if not np.isfinite(self.peak_time_s):
            raise ValueError("Ricker peak time must be finite")

    def __call__(self, time_s: float) -> float:
        phase_squared = (np.pi * self.frequency_hz * (time_s - self.peak_time_s)) ** 2
        return float((1 - 2 * phase_squared) * np.exp(-phase_squared))


@dataclass(frozen=True)
class PointForce:
    stencil: PointStencil
    peak_force_n: NDArray[np.float64]  # Cartesian force vector, including amplitude.
    wavelet: Ricker

    def add_to(self, load: NDArray[np.float64], time_s: float) -> None:
        self.stencil.add_load(load, self.wavelet(time_s) * self.peak_force_n)


@dataclass(frozen=True)
class AcousticPointSource:
    """Scalar load in the potential equation, with amplitude in cubic metres."""

    stencil: PointStencil
    amplitude_m3: float
    wavelet: Ricker

    def add_to(self, load: NDArray[np.float64], time_s: float) -> None:
        self.stencil.add_load(load, np.array([self.wavelet(time_s) * self.amplitude_m3]))


@dataclass(frozen=True)
class Receiver:
    station_id: str
    stencil: PointStencil


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str


DISPLACEMENT_CHANNELS = (Channel("ux", "m"), Channel("uy", "m"), Channel("uz", "m"))
PRESSURE_CHANNELS = (Channel("pressure", "Pa"),)


@dataclass(frozen=True)
class Seismogram:
    time_s: NDArray[np.float64]
    station_ids: tuple[str, ...]
    channels: tuple[Channel, ...]
    values: NDArray[np.float64]  # (sample, station, channel)

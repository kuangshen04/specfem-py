import csv
from pathlib import Path

from .acquisition import Seismogram


def write_seismogram(path: Path, seismogram: Seismogram) -> None:
    shape = (len(seismogram.time_s), len(seismogram.station_ids), len(seismogram.channels))
    if seismogram.values.shape != shape:
        raise ValueError("seismogram values must match its sample, station and channel axes")
    columns = [f"{channel.name}_{channel.unit}" for channel in seismogram.channels]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "station_id", *columns])
        for time, samples in zip(seismogram.time_s, seismogram.values):
            for station, sample in zip(seismogram.station_ids, samples):
                writer.writerow([time, station, *sample])

# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset utilities."""

from typing import Any, Mapping, Sequence, Union

import numpy as np
import xarray


TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.

TargetLeadTimes = Union[TimedeltaLike, Sequence[TimedeltaLike], slice]  # with TimedeltaLike as its start and stop.

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"
_DERIVED_VARS = {
    DAY_PROGRESS,
    f"{DAY_PROGRESS}_sin",
    f"{DAY_PROGRESS}_cos",
    YEAR_PROGRESS,
    f"{YEAR_PROGRESS}_sin",
    f"{YEAR_PROGRESS}_cos",
}
TISR = "toa_incident_solar_radiation"


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
    """Computes year progress for times in seconds.

    Args:
      seconds_since_epoch: Times in seconds since the "epoch" (the point at which
        UNIX time starts).

    Returns:
      Year progress normalized to be in the [0, 1) interval for each time point.
    """

    # Start with the pure integer division, and then float at the very end.
    # We will try to keep as much precision as possible.
    years_since_epoch = seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
    # Note depending on how these ops are down, we may end up with a "weak_type"
    # which can cause issues in subtle ways, and hard to track here.
    # In any case, casting to float32 should get rid of the weak type.
    # [0, 1.) Interval.
    return np.mod(years_since_epoch, 1.0).astype(np.float32)


def get_day_progress(
    seconds_since_epoch: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Computes day progress for times in seconds at each longitude.

    Args:
      seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
        point at which UNIX time starts).
      longitude: 1D array of longitudes at which day progress is computed.

    Returns:
      2D array of day progress values normalized to be in the [0, 1) inverval
        for each time point at each longitude.
    """

    # [0.0, 1.0) Interval.
    day_progress_greenwich = np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY

    # Offset the day progress to the longitude of each point on Earth.
    longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
    day_progress = np.mod(day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0)
    return day_progress.astype(np.float32)


def featurize_progress(name: str, dims: Sequence[str], progress: np.ndarray) -> Mapping[str, xarray.Variable]:
    """Derives features used by ML models from the `progress` variable.

    Args:
      name: Base variable name from which features are derived.
      dims: List of the output feature dimensions, e.g. ("day", "longitude").
      progress: Progress variable values.

    Returns:
      Dictionary of xarray variables derived from the `progress` values. It
      includes the original `progress` variable along with its sin and cos
      transformations.

    Raises:
      ValueError if the number of feature dimensions is not equal to the number
        of data dimensions.
    """
    if len(dims) != progress.ndim:
        raise ValueError(
            f"Number of feature dimensions ({len(dims)}) must be equal to the"
            f" number of data dimensions: {progress.ndim}."
        )
    progress_phase = progress * (2 * np.pi)
    return {
        name: xarray.Variable(dims, progress),
        name + "_sin": xarray.Variable(dims, np.sin(progress_phase)),
        name + "_cos": xarray.Variable(dims, np.cos(progress_phase)),
    }


def get_seconds_since_epoch(time_sequence: xarray.DataArray) -> np.ndarray:
    """Computes seconds since epoch from `data` in place if missing."""
    # Note `time_sequence.astype("datetime64[s]").astype(np.int64)`
    # does not work as xarrays always cast dates into nanoseconds!
    return time_sequence.data.astype("datetime64[s]").astype(np.int64)


def add_derived_vars(data: xarray.Dataset) -> None:
    """Adds year and day progress features to `data` in place if missing.

    Args:
      data: Xarray dataset to which derived features will be added.

    Raises:
      ValueError if `time` or `lon` are not in `data` coordinates.
    """

    for coord in ("time", "longitude"):
        if coord not in data.coords:
            raise ValueError(f"'{coord}' must be in `data` coordinates.")

    # Compute seconds since epoch.
    seconds_since_epoch = get_seconds_since_epoch(data.coords["time"])
    batch_dim = ("batch",) if "batch" in data.dims else ()

    # Add year progress features if missing.
    if YEAR_PROGRESS not in data.data_vars:
        year_progress = get_year_progress(seconds_since_epoch)
        data.update(
            featurize_progress(
                name=YEAR_PROGRESS,
                dims=batch_dim + ("time",),
                progress=year_progress,
            )
        )

    # Add day progress features if missing.
    if DAY_PROGRESS not in data.data_vars:
        longitude_coord = data.coords["longitude"]
        day_progress = get_day_progress(seconds_since_epoch, longitude_coord.data)
        data.update(
            featurize_progress(
                name=DAY_PROGRESS,
                dims=batch_dim + ("time",) + longitude_coord.dims,
                progress=day_progress,
            )
        )


def add_tisr_var(data: xarray.Dataset) -> None:
    """Adds TISR feature to `data` in place if missing.

    Args:
      data: Xarray dataset to which TISR feature will be added.

    Raises:
      ValueError if `time`, 'latitude', or `lon` are not in `data` coordinates.
    """
    from src.datamodules.utils import solar_radiation

    if TISR in data.data_vars:
        return

    for coord in ("time", "latitude", "longitude"):
        if coord not in data.coords:
            raise ValueError(f"'{coord}' must be in `data` coordinates.")

    # Remove `batch` dimension of size one if present. An error will be raised if
    # the `batch` dimension exists and has size greater than one.
    data_no_batch = data.squeeze("batch") if "batch" in data.dims else data

    tisr = solar_radiation.get_toa_incident_solar_radiation_for_xarray(data_no_batch, use_jit=True)

    if "batch" in data.dims:
        tisr = tisr.expand_dims("batch", axis=0)

    data.update({TISR: tisr})

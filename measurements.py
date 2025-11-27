import logging
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from numpy.fft import fft, fftfreq, rfft, rfftfreq
from functions import window, remove_offset
from enum import Enum


class Domain(Enum):
    Time = 0
    Frequency = 1
    Both = 2


class MeasurementType(Enum):
    REF = 1
    SAM = 2


class Measurement:
    filepath = None
    meas_time = None
    meas_type = None
    sample_name = None
    position = (None, None)
    window_applied = False
    offset_corrected = False
    identifier = None

    def __init__(self, filepath=None):
        self.filepath = filepath

        self._data_fd, self._data_td = None, None

        self._set_metadata()

    def __repr__(self):
        return str(self.filepath)

    def _extract_position(self):
        fp_stem = str(self.filepath.stem)

        matches = re.findall(r"(-?\d+\.\d+|-?\d+) mm", fp_stem)
        positions = [0.000 if np.isclose(float(val), 0) else float(val) for val in matches]
        l_diff = 2 - len(positions)
        if l_diff > 0:
            positions.extend(l_diff * [0.0])
        positions = tuple(positions)

        return positions

    def _set_metadata(self):
        # set time
        date_string = str(self.filepath.stem)[:25]
        self.meas_time = datetime.strptime(date_string, "%Y-%m-%dT%H-%M-%S.%f")

        # set sample name
        dir_1above, dir_2above = self.filepath.parents[0], self.filepath.parents[1]
        if ("sam" in dir_1above.stem.lower()) or ("ref" in dir_1above.stem.lower()):
            self.sample_name = dir_2above.stem
        else:
            self.sample_name = dir_1above.stem

        # set measurement type
        if "ref" in str(self.filepath.stem).lower():
            self.meas_type = MeasurementType(1)
        else:
            self.meas_type = MeasurementType(2)

        # set position
        self.position = self._extract_position()

        # set identifier
        self.identifier = int((self.meas_time-datetime.min).total_seconds() * 1e6)

    def get_data_td(self):
        if self._data_td is None:
            self._data_td = np.loadtxt(self.filepath)

        return self._data_td

    def get_data_fd(self, reversed_time=True):
        if self._data_fd is not None:
            return self._data_fd

        data_td = self.get_data_td()
        t, y = data_td[:, 0], data_td[:, 1]

        if reversed_time:
            y = np.flip(y)

        dt = float(np.mean(np.diff(t)))
        freqs, data_fd = rfftfreq(n=len(t), d=dt), rfft(y)

        self._data_fd = np.array([freqs, data_fd]).T

        return self._data_fd

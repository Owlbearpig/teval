import logging
import re
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from datetime import datetime
from numpy.fft import fft, fftfreq, rfft, rfftfreq
from teval.functions import window
from enum import Enum


class Domain(Enum):
    TimeDomain = 0
    FrequencyDomain = 1
    Both = 2


class MeasurementType(Enum):
    REF = 1
    SAM = 2
    OTHER = 3


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
        positions = [0.0 if np.isclose(float(val), 0) else float(val) for val in matches]

        l_diff = 2 - len(positions)
        if l_diff > 0:
            positions.extend(l_diff * [0.0])

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
        elif "sam" in str(self.filepath.stem).lower():
            self.meas_type = MeasurementType(2)
        else:
            self.meas_type = MeasurementType(3)

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


def select_measurements(measurements, keywords, case_sensitive=True, match_exact=False):
    if not case_sensitive:
        keywords = [keyword.lower() for keyword in keywords]

    selected = []
    for measurement in measurements:
        dirs = measurement.filepath.parents[0].parts
        if match_exact:
            for dir_ in dirs:
                if any([keyword == dir_ for keyword in keywords]):
                    selected.append(measurement)
                    break
        elif all([keyword in str(measurement) for keyword in keywords]):
            selected.append(measurement)

    if len(selected) == 0:
        exit("No files found; exiting")

    ref_cnt, sam_cnt = 0, 0
    for selected_measurement in selected:
        if selected_measurement.meas_type == "sam":
            sam_cnt += 1
        elif selected_measurement.meas_type == "ref":
            ref_cnt += 1
    print(f"Number of reference and sample measurements in selection: {ref_cnt}, {sam_cnt}")

    selected.sort(key=lambda x: x.meas_time)

    print("Time between first and last measurement: ", selected[-1].meas_time - selected[0].meas_time)

    sams = [x for x in selected if x.meas_type == "sam"]
    refs = [x for x in selected if x.meas_type == "ref"]

    return refs, sams

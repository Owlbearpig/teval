import re
import numpy as np
from datetime import datetime
from pathlib import Path

meas_id_func = lambda meas_datetime: int((meas_datetime - datetime.min).total_seconds() * 1e6)
timestamp2id = lambda timestamp_str: meas_id_func(datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%f"))

class Measurement:
    filepath = Path()
    meas_time = None
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
        date_string = str(self.filepath.stem)[:26]
        self.meas_time = datetime.strptime(date_string, "%Y-%m-%dT%H-%M-%S.%f")

        dir_1above, dir_2above = self.filepath.parents[0], self.filepath.parents[1]
        if ("sam" in dir_1above.stem.lower()) or ("ref" in dir_1above.stem.lower()):
            self.sample_name = dir_2above.stem
        else:
            self.sample_name = dir_1above.stem

        self.position = self._extract_position()

        self.identifier = meas_id_func(self.meas_time)

    def get_data(self):
        if self._data_td is None:
            self._data_td = np.loadtxt(self.filepath)

        return self._data_td

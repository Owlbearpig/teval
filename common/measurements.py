import re
import numpy as np
from datetime import datetime
from pathlib import Path
from common.components import ComponentBase
from common.functions import avg_data_array, f_axis_idx_map
from enum import Enum
from traitlets import Enum as TEnum

meas_id_func = lambda meas_datetime: int((meas_datetime - datetime.min).total_seconds() * 1e6)
timestamp2id = lambda timestamp_str: meas_id_func(datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%f"))

class Averaging(Enum):
    only_avg = "Average only"
    only_individual = "Individual measurements only"
    both = "Average and individual measurements"

class DataSetBase(ComponentBase):
    freq_range = None
    averaging = TEnum(Averaging, Averaging.only_avg)

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.measurement_selector = dataset.measurement_selector

    def __enter__(self, *args):
        if self.settings is not None:
            self.settings.load_configuration(self)

    def __exit__(self, *args):
        if self.settings is not None:
            self.settings.save_configuration(self)

    @property
    def measurements(self):
        return self.dataset.measurements

    @property
    def settings(self):
        return self.dataset.settings

    @property
    def freq_axis(self):
        f_axis = self.dataset.freq_axis[self.freq_range_idx]
        return f_axis

    @property
    def freq_range_idx(self):
        return f_axis_idx_map(self.dataset.freq_axis, self.freq_range)

    def format_meas_dict(self, meas_list, data_array: np.ndarray):
        if data_array.shape[1] != len(self.freq_axis):
            if data_array.ndim == 2:
                data_array = data_array[:, self.freq_range_idx]
            elif data_array.ndim == 3:
                data_array = data_array[:, self.freq_range_idx, :]

        if data_array.ndim == 2:
            freq_tile = np.tile(self.freq_axis, (len(meas_list), 1))
            data_array = np.stack((freq_tile, data_array, np.zeros_like(freq_tile)), axis=2)

        if len(meas_list) == 1:
            return {meas_list[0]: data_array[0]}

        average_y_meas = {"Average": avg_data_array(data_array)}
        individual_y_meas = {
            meas: data_array[meas_idx]
            for meas_idx, meas in enumerate(meas_list)
        }
        match self.averaging:
            case Averaging.only_avg:
                return average_y_meas
            case Averaging.only_individual:
                return individual_y_meas
            case Averaging.both:
                return {**average_y_meas, **individual_y_meas}

class Measurement:
    filepath = Path()
    meas_time = None
    sample_name = None
    position = (None, None)
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

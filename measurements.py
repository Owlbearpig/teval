import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from teval.consts import data_dir
from numpy.fft import fft, fftfreq
from teval.functions import window
from enum import Enum


class MeasurementType(Enum):
    REF = 1
    SAM = 2
    OTHER = 3


class Measurement:
    filepath = None

    def __init__(self, data_td=None, meas_type=None, filepath=None, post_process_config=None):
        self.filepath = filepath
        self.meas_time = None
        self.meas_type = None
        self.sample_name = None
        self.position = [None, None]

        if post_process_config is None:
            from .consts import post_process_config

        self.post_process_config = post_process_config
        self._data_fd, self._data_td = None, data_td
        self.pre_process_done = False

        self._set_metadata(meas_type)

    def __repr__(self):
        return str(self.filepath)

    def _set_metadata(self, meas_type=None):
        if meas_type is not None:
            self.meas_type = meas_type
            return

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
        str_splits = str(self.filepath).split("_")
        x = float(str_splits[-2].split(" mm")[0])
        y = float(str_splits[-1].split(" mm")[0])
        if np.isclose(x, 0):
            x = 0.0
        if np.isclose(y, 0):
            y = 0.0

        self.position = [x, y]

    def do_preprocess(self, force=False):
        if self.pre_process_done and not force:
            return

        if self.post_process_config["sub_offset"]:
            self._data_td[:, 1] -= np.mean(self._data_td[:10, 1])
        if self.post_process_config["en_windowing"]:
            self._data_td = window(self._data_td)

        self.pre_process_done = True

    def get_data_td(self, get_raw=False):
        if get_raw:
            return np.loadtxt(self.filepath)

        if self._data_td is None:
            self._data_td = np.loadtxt(self.filepath)

        self.do_preprocess()

        return self._data_td

    def get_data_fd(self, pos_freqs_only=True, reversed_time=True):
        if self._data_fd is not None:
            return self._data_fd

        data_td = self.get_data_td()
        t, y = data_td[:, 0], data_td[:, 1]

        if reversed_time:
            y = np.flip(y)

        dt = float(np.mean(np.diff(t)))
        freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y)

        if pos_freqs_only:
            pos_slice = freqs >= 0
            self._data_fd = np.array([freqs[pos_slice], data_fd[pos_slice]]).T
        else:
            self._data_fd = np.array([freqs, data_fd]).T

        return self._data_fd

    def get_data_both_domains(self):
        return self.get_data_td(), self.get_data_fd()


def get_all_measurements(post_process=None, data_dir_=None):
    measurements = []

    if data_dir_ is not None:
        glob = data_dir_.glob("**/*.txt")
    else:
        glob = data_dir.glob("**/*.txt")

    for file_path in glob:
        if file_path.is_file():
            try:
                measurements.append(Measurement(filepath=file_path, post_process_config=post_process))
            except ValueError:
                print(f"Skipping, not a measurement: {file_path}")

    return measurements


def avg_data(measurements):
    data_0 = measurements[0].get_data_td()
    t = data_0[:, 0]

    y_arrays = []
    for measurement in measurements:
        data_td = measurement.get_data_td()
        y_arrays.append(data_td[:, 1])

    return np.array([t, np.mean(y_arrays, axis=0)]).T


def select_measurements(keywords, case_sensitive=True, post_process=None, match_exact=False):
    measurements = get_all_measurements(post_process=post_process)

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


def get_avg_measurement(keywords=("GaAs", "Wafer", "25", "2021_08_24"), pp_config=None):
    if pp_config is None:
        pp_config = {"sub_offset": True, "en_windowing": True}

    refs, sams = select_measurements(keywords)

    avg_ref = Measurement(data_td=avg_data(refs), meas_type="ref", post_process_config=pp_config)
    avg_sam = Measurement(data_td=avg_data(sams), meas_type="sam", post_process_config=pp_config)

    return avg_ref, avg_sam

import os
import shutil
import numpy as np
from pathlib import Path


def cache_dir(data_dir):
    return Path(data_dir) / "_cache"


def remove_cache_dir(data_dir):
    # the whole folder, so files of older cache formats go as well
    path = cache_dir(data_dir)
    if path.exists():
        shutil.rmtree(path)
    return path

def _generate_id_map(measurements):
    # measurement index sorted by identifier (Higher identifier = later measurement time)
    # {63912383001176544: 0, 63912383010246552: 1, 63912383019536552: 2, ...}
    sorted_measurements = sorted(measurements, key=lambda x: x.identifier)
    ids = [id_.identifier for id_ in sorted_measurements]
    meas_idx_list = list(range(len(measurements)))

    return dict(zip(ids, meas_idx_list))


def _generate_filepath_map(measurements):
    filepath_map = {}
    for meas in measurements:
        filepath_map[meas.filepath] = meas

    return filepath_map

class DatasetCache:

    def __init__(self, dataset, new_measurements, data_dir, signal_carrier=None):
        self.dataset = dataset
        all_measurements = new_measurements["all"]

        self.signal_carrier = signal_carrier
        self.coord_map_key_func = lambda position_tuple: "_".join([f"{val:.3f}" for val in position_tuple])
        self.coord_map = self._generate_coord_map(all_measurements)

        self.id_map = _generate_id_map(all_measurements)
        self.filepath_map = _generate_filepath_map(all_measurements)

        self.data_dir = data_dir
        self.path = None
        self.time_axis = None

        self.raw_data_td = self._set_cache_data(all_measurements)


    def _generate_coord_map(self, measurements):
        coord_map = {}
        for meas in measurements:
            meas_position = meas.position
            k = self.coord_map_key_func(meas_position)
            if k in coord_map:
                coord_map[k].append(meas)
            else:
                coord_map[k] = [meas]

        return coord_map

    def get_path(self):
        self.path = cache_dir(self.data_dir)
        self.path.mkdir(parents=True, exist_ok=True)

        return self.path

    def clear_cache(self):
        try:
            self.path = remove_cache_dir(self.data_dir)
        except OSError as e:
            self.dataset.logger.warning(f"Could not remove the cache directory: {e}")
        else:
            self.raw_data_td = None
            self.time_axis = None

            self.dataset.logger.info(f"Cleared {self.path}")
            if self.signal_carrier is not None:
                self.signal_carrier.progress_signal.emit(0)
                self.signal_carrier.initialization_complete_signal.emit("is_initialized", False)

    def _set_cache_data(self, measurements):
        # make cache (npy) if it does not already exist
        path = self.get_path()

        sample_data_td = measurements[0].get_data()
        self.time_axis = sample_data_td[:, 0].real
        td_cache_shape = (len(measurements), len(sample_data_td[:, 1]))

        cache_file = path / "_td_cache.npy"
        try:
            data_td = np.load(str(cache_file))
            shape_match = (data_td.shape == td_cache_shape)
            if not shape_match:
                self.dataset.logger.error(f"Data <-> cache shape mismatch. Reloading data:")
                raise FileNotFoundError
        except (FileNotFoundError, ValueError, EOFError, OSError) as e:
            if not isinstance(e, FileNotFoundError):
                self.dataset.logger.warning(f"Unreadable cache file {cache_file} ({e}), rebuilding it")
            data_td = np.zeros(td_cache_shape, dtype=sample_data_td.dtype)

            self.dataset.logger.info(f"Saving {len(measurements)} measurements as npy")
            for meas_idx, meas in enumerate(measurements):
                idx = self.id_map[meas.identifier]
                data_td[idx] = meas.get_data()[:, 1]
                if self.signal_carrier is not None:
                    progress = (1 + meas_idx) / len(measurements)
                    self.signal_carrier.progress_signal.emit(progress)

            # written under a temporary name so an interrupted save can't leave a truncated cache behind
            tmp_file = path / "_td_cache.tmp.npy"
            np.save(str(tmp_file), data_td)
            os.replace(tmp_file, cache_file)

        return data_td


import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging


def _generate_id_map(measurements):
    # measurement index sorted by identifier (Higher identifier = later measurement time)
    # {63912383001176544: 0, 63912383010246552: 1, 63912383019536552: 2, ...}
    sorted_measurements = sorted(measurements, key=lambda x: x.identifier)
    ids = [id_.identifier for id_ in sorted_measurements]
    meas_idx_list = list(range(len(measurements)))

    return dict(zip(ids, meas_idx_list))


class DatasetCache:

    def __init__(self, measurements, data_dir, signal_carrier=None):
        self.signal_carrier = signal_carrier
        self.coord_map_key_func = lambda position_tuple: "_".join([f"{val:.3f}" for val in position_tuple])
        self.coord_map = self._generate_coord_map(measurements)

        self.id_map = _generate_id_map(measurements)

        self.data_dir = data_dir
        self.raw_data_td, self.raw_data_fd = self._make_cache(measurements)
        self.path = None


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
        self.path = Path(self.data_dir / "_cache")
        self.path.mkdir(parents=True, exist_ok=True)

        return self.path

    def clear_cache(self):
        if self.path is None:
            self.path = Path(self.data_dir / "_cache")

        if self.path.exists():
            td_file = self.path / "_td_cache.npy"
            fd_file = self.path / "_fd_cache.npy"

            if td_file.exists():
                td_file.unlink()
            if fd_file.exists():
                fd_file.unlink()

            try:
                self.path.rmdir()
            except OSError as e:
                logging.warning(f"Could not remove directory {self.path}: {e}")

            self.raw_data_td = None
            self.raw_data_fd = None

            logging.info(f"Cleared {self.path}")
            if self.signal_carrier is not None:
                self.signal_carrier.progress_signal.emit(0)
                self.signal_carrier.initialization_complete_signal.emit("is_initialized", False)

    def _make_cache(self, measurements):
        # make cache (npy) if it does not already exist
        path = self.get_path()

        y_td, y_fd = measurements[0].get_data_td(), measurements[0].get_data_fd()
        td_cache_shape = (len(measurements), *y_td.shape)
        fd_cache_shape = (len(measurements), *y_fd.shape)

        try:
            data_td = np.load(str(path / "_td_cache.npy"))
            data_fd = np.load(str(path / "_fd_cache.npy"))
            shape_match = (data_td.shape == td_cache_shape) * (data_fd.shape == fd_cache_shape)
            if not shape_match:
                logging.error("Data <-> cache shape mismatch. Reloading data:")
                raise FileNotFoundError
        except FileNotFoundError:
            data_td = np.zeros(td_cache_shape, dtype=y_td.dtype)
            data_fd = np.zeros(fd_cache_shape, dtype=y_fd.dtype)

            logging.info(f"Saving {len(measurements)} measurements as npy")
            for meas_idx, meas in enumerate(measurements):
                idx = self.id_map[meas.identifier]
                data_td[idx], data_fd[idx] = meas.get_data_td(), meas.get_data_fd()
                if self.signal_carrier is not None:
                    progress = (1 + meas_idx) / len(measurements)
                    self.signal_carrier.progress_signal.emit(progress)

            np.save(str(path / "_td_cache.npy"), data_td)
            np.save(str(path / "_fd_cache.npy"), data_fd)

        return data_td, data_fd


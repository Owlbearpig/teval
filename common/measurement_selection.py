from __future__ import annotations
from dataclasses import dataclass
import threading
import logging
import numpy as np
from common.components import ComponentBase, action
from common.settings import Settings
from common.traits import MultiPathSelection, ValueRange, MultiPathClass, StrListSelection, StrList
from common.units import Q_
from common.default_appsettings import Dist
from traitlets import Enum as TEnum, Unicode, Bool, Int, Instance, observe
from enum import Enum
from common.measurements import timestamp2id, Measurement
from types import TracebackType


def get_coordinate_line(measurements, x=None, y=None):
    if not measurements:
        return []

    if isinstance(x, Q_):
        x = x.magnitude
    if isinstance(y, Q_):
        y = y.magnitude

    if x is not None:
        all_x = np.array([m.position[0] for m in measurements])
        closest_x = all_x[np.argmin(np.abs(all_x - x))]

        line_measurements = [m for m in measurements if m.position[0] == closest_x]

        line_measurements.sort(key=lambda m: m.position[1])

    else:
        all_y = np.array([m.position[1] for m in measurements])
        closest_y = all_y[np.argmin(np.abs(all_y - y))]

        line_measurements = [m for m in measurements if m.position[1] == closest_y]

        line_measurements.sort(key=lambda m: m.position[0])

    return line_measurements

class SelectionCriterionEnum(Enum):
    file_selection = "File selection"
    selected_timestamp = "Timestamp"
    selected_point = "Point"
    string_search = "String"

class ReferenceSelection(Enum):
    file_selection = "File selection"
    max_amp_measurement = "Maximum amplitude measurement"
    closest_distance = "Closest distance"
    fix_ref = "Use fixed index reference"


@dataclass(frozen=True)
class Selection:
    sams: tuple[Measurement, ...]
    refs: tuple[Measurement, ...]

    @property
    def ref_map(self):
        d = dict(zip(self.sams, self.refs))
        return lambda meas: d[meas] if isinstance(meas, Measurement) else meas

    @property
    def label(self) -> str:
        meas = self.sams or self.refs
        if not meas:
            return "empty"
        extra = f" (+{len(meas) - 1})" if len(meas) > 1 else ""
        return f"{meas[0].filepath.name}{extra}"

    def info(self) -> str:
        lines = [f"Samples: {len(self.sams)}",
                 f"References: {len(self.refs)} ({len(set(self.refs))} unique)"]
        if self.sams:
            lines.append(f"First sample: {self.sams[0].filepath.name}")
            if len(self.sams) > 1:
                lines.append(f"Last sample: {self.sams[-1].filepath.name}")
        return "\n".join(lines)

    __repr__ = label.fget


class SelectionQueue(ComponentBase):
    queued_selections = StrListSelection(group="Queued selections", read_only=True, priority=1, fullwidth=True)
    selection_info = Unicode("", read_only=True).tag(group="Selection info", en_save=False,
                                                     priority=2, fullwidth=True, disable_label=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._queue: dict[str, Selection] = {}
        self._counter = 0
        self._lock = threading.RLock()
        self.set_trait("queued_selections", StrList())
        self.queued_selections.observe(self.on_list_selection, "selected_item")

    def _update_listing(self):
        self.queued_selections.items = list(self._queue)

    def _refresh_info(self):
        sel = self._queue.get(self.queued_selections.selected_item)
        self.set_trait("selection_info", sel.info() if sel else "")

    def on_list_selection(self, change):
        with self._lock:
            self._refresh_info()

    def get_selection(self, label):
        with self._lock:
            return self._queue.get(label)

    def append(self, selection):
        with self._lock:
            self._counter += 1
            self._queue[f"#{self._counter} {selection.label}"] = selection
            self._update_listing()

    def pop_next(self):
        with self._lock:
            if not self._queue:
                return None
            selection = self._queue.pop(next(iter(self._queue)))
            self._update_listing()
            self._refresh_info()
            return selection

    def clear(self):
        with self._lock:
            self._queue.clear()
            self._update_listing()
            self._refresh_info()

    def __len__(self):
        return len(self._queue)


class MeasurementSelection(ComponentBase):
    measurement_selection_grp = "Measurement selection"
    selection_criterion = TEnum(SelectionCriterionEnum,
                                SelectionCriterionEnum.file_selection).tag(name="Select measurement by",
                                                                           group=measurement_selection_grp,
                                                                           priority=-2)
    sel_point = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(name="Selected point (x, y)",
                                                                             group=measurement_selection_grp)
    sel_timestamp = Unicode("").tag(name="Selected timestamp", group=measurement_selection_grp)
    string_match = Unicode("").tag(name="Filter string", group=measurement_selection_grp)
    selected_sam_cnt = Unicode("", read_only=True).tag(name="Selected sample measurements", priority=2000,
                                                       group=measurement_selection_grp)

    reference_matching_grp = "Reference matching"
    ref_sel_criterion = TEnum(ReferenceSelection,
                              ReferenceSelection.file_selection).tag(name="Reference matching criterion",
                                                                          group=reference_matching_grp,
                                                                          priority=-2)
    dist_func = TEnum(Dist, default_value=Dist.Time).tag(priority=1000, name="Measurement distance function",
                                                         group=reference_matching_grp)
    fix_ref_idx = Int(0, min=-1, group=reference_matching_grp).tag(name="Fixed reference index")
    selected_ref_cnt = Unicode("",read_only=True).tag(name="Selected references", priority=2001,
                                                      group=reference_matching_grp)
    direct_match = Bool(False, read_only=True,
                        help="Appends or slices reference file selection if the count is "
                             "different from the measurement file selection"
                        ).tag(name="Direct file selection match", priority=2000, group=reference_matching_grp)

    selection_queue = Instance(SelectionQueue, allow_none=True)

    reference_paths = MultiPathSelection().tag(fullwidth=False, group="Direct reference file selection", combine=True)
    sample_paths = MultiPathSelection().tag(fullwidth=False, group="Direct sample file selection", combine=True)


    def __init__(self, dataset, en_queue=True, **kwargs):
        super().__init__(**kwargs)

        self.dataset = dataset

        ref_filenames = [f"{meas.filepath.name}" for meas in self.dataset.measurements["refs"]]
        sam_filenames = [f"{meas.filepath.name}" for meas in self.dataset.measurements["sams"]]

        self.reference_paths = MultiPathClass(root_path=self.dataset.data_path, shown_filenames=ref_filenames)
        self.sample_paths = MultiPathClass(root_path=self.dataset.data_path, shown_filenames=sam_filenames)

        if en_queue:
            self.selection_queue = SelectionQueue(object_name="Selection queue")

    def set_observers(self):
        self.dataset.observe(self.update_fileselection, "measurements")
        self.dataset.observe(self.update_sel_cnt_info, "measurements")

        reference_sel_names = self.trait_names(group=self.reference_matching_grp)
        self.observe(self.update_sel_cnt_info, names=reference_sel_names)

        measurement_sel_names = self.trait_names(group=self.measurement_selection_grp)
        self.observe(self.update_sel_cnt_info, names=measurement_sel_names)

        self.reference_paths.observe(self.ref_file_sel_cnt, names="selected_paths")
        self.sample_paths.observe(self.sam_file_sel_cnt, names="selected_paths")

    def update_fileselection(self, change):
        new_measurements = change["new"]
        root_path = self.dataset.data_path

        ref_filenames = [f"{meas.filepath.name}" for meas in new_measurements["refs"]]
        sam_filenames = [f"{meas.filepath.name}" for meas in new_measurements["sams"]]

        self.reference_paths = MultiPathClass(root_path=root_path,
                                              selected_paths=self.reference_paths.selected_paths,
                                              shown_filenames=ref_filenames)
        self.sample_paths = MultiPathClass(root_path=root_path,
                                           selected_paths=self.sample_paths.selected_paths,
                                           shown_filenames=sam_filenames)

        self.reference_paths.observe(self.update_sel_cnt_info, names="selected_paths")
        self.sample_paths.observe(self.update_sel_cnt_info, names="selected_paths")

    @property
    def measurements(self):
        return self.dataset.measurements

    @property
    def cache(self):
        return self.dataset.cache

    @property
    def selected_measurements(self):
        return self.get_selected_measurements()

    @property
    def sam_ref_meas_map(self):
        meas_list = self.selected_measurements
        ref_list = self.get_matching_refs(meas_list)

        dict_map = {meas: ref_list[i] for i, meas in enumerate(meas_list)}
        return lambda meas: dict_map[meas] if isinstance(meas, Measurement) else meas

    @action("Add selection to queue", enable_checker=lambda inst: inst.selection_queue is not None, priority=1)
    def queue_selection(self):
        selection = self.get_selection()
        if selection is None:
            self.dataset.logger.warning("Nothing to queue")
            return
        self.selection_queue.append(selection)
        sam_s = "" if len(selection.sams) <= 1 else "s"
        sel_s = "" if len(self.selection_queue) <= 1 else "s"
        msg = f"Queued {len(selection.sams)} measurement{sam_s}, {len(self.selection_queue)} selection{sel_s} in queue"
        self.dataset.logger.info(msg)

    @action("Clear queue", enable_checker=lambda inst: inst.selection_queue is not None, priority=2)
    def clear_queue(self):
        queue_len = len(self.selection_queue)
        self.selection_queue.clear()
        self.dataset.logger.info(f"Cleared {queue_len} {"selection" + "s" * (queue_len - 1)} from queue")

    def get_selection(self, must_match=True):
        sams = self.get_selected_measurements()
        if not sams:
            self.dataset.logger.warning("No measurements selected")
            return None
        refs = self.get_matching_refs(sams)
        selection = Selection(tuple(sams), tuple(refs))

        if not must_match:
            return selection

        if len(refs) != len(sams):
            self.dataset.logger.warning("Could not resolve a reference for every sample")
            return None
        else:
            return selection

    def update_sel_cnt_info(self, change):
        change_name = change["name"]
        if change_name in ["selected_ref_cnt", "selected_sam_cnt"]:
            return

        selected_measurements = self.selected_measurements
        if not selected_measurements and (self.ref_sel_criterion != ReferenceSelection.file_selection):
            return
        matching_refs = self.get_matching_refs(selected_measurements)

        self.set_trait("selected_ref_cnt", f"{len(matching_refs)}")
        self.set_trait("selected_sam_cnt", f"{len(selected_measurements)}")

    def ref_file_sel_cnt(self, change=None):
        if self.ref_sel_criterion != ReferenceSelection.file_selection:
            return
        rm_len = len(self.get_matching_refs(self.selected_measurements))
        self.set_trait("selected_ref_cnt", f"{rm_len}")

    def sam_file_sel_cnt(self, change=None):
        if self.selection_criterion != SelectionCriterionEnum.file_selection:
            return
        self.set_trait("selected_sam_cnt", f"{len(self.selected_measurements)}")

    def get_measurements_from_point(self, x, y, return_single=False):
        if self.cache is None:
            self.dataset.logger.info("Cache not loaded, check dataset path")
            return []

        if isinstance(x, Q_):
            x = x.magnitude
        if isinstance(y, Q_):
            y = y.magnitude
        pnt = (x, y)

        try:
            key = self.cache.coord_map_key_func(pnt)
            found_meas_list = self.cache.coord_map[key]
        except KeyError:
            all_points = np.array(self.dataset.shape_properties["all_points"])
            dist_diff_squared = np.abs(all_points - pnt) ** 2

            closest_pnt_idx = np.argmin(np.sum(dist_diff_squared, axis=1))

            key = self.cache.coord_map_key_func(all_points[closest_pnt_idx])
            found_meas_list = self.cache.coord_map[key]

        return found_meas_list[0] if return_single else found_meas_list

    def get_measurements_from_timestamp(self, timestamp_str=""):
        if not timestamp_str:
            self.dataset.logger.warning("No timestamp set. Returning first measurement")
            return [self.measurements["all"][0]]
        self.dataset.logger.info(f"Selecting measurement by timestamp {timestamp_str}")
        meas_id_ = timestamp2id(timestamp_str)

        found_meas_list = []
        for meas in self.measurements["all"]:
            if meas.identifier == meas_id_:
                found_meas_list.append(meas)

        if not found_meas_list:
            self.dataset.logger.warning(f"No measurement with timestamp: {timestamp_str} "
                                        f"(id: {meas_id_}) found in dataset")

        return found_meas_list

    def get_measurements_from_string(self, string=""):
        if not string:
            self.dataset.logger.warning("No filter string set. Returning first measurement")
            return [self.measurements["all"][0]]

        found_meas_list = []
        for meas in self.measurements["all"]:
            if string in meas.filepath.name:
                found_meas_list.append(meas)

        if not found_meas_list:
            self.dataset.logger.warning(f"No measurement containing the string {string} in the filename found in dataset")

        return found_meas_list

    def get_consecutive_meas(self, meas_):
        # measurements with same position as meas_ sampled without interruption (compared to avg meas time)
        coord_map_key = self.cache.coord_map_key_func(meas_.position)
        meas_at_pos = self.cache.coord_map[coord_map_key]
        if len(meas_at_pos) == 1:
            return meas_at_pos

        meas_idx0 = meas_at_pos.index(meas_)
        max_dist = 2*self.dataset.mean_time_diff

        time_diff = np.diff([meas.meas_time for meas in meas_at_pos])
        time_diff_sec = [t_diff.total_seconds() for t_diff in time_diff]
        jump_idx_list = np.where(time_diff_sec > max_dist)[0]

        interval_idx = np.digitize(meas_idx0, jump_idx_list, right=True)
        if interval_idx == 0:
            meas_idx_range = np.arange(0, jump_idx_list[0]+1)
        elif interval_idx == len(jump_idx_list):
            meas_idx_range = np.arange(jump_idx_list[-1]+1, len(meas_at_pos))
        else:
            meas_idx_range = np.arange(jump_idx_list[interval_idx-1]+1, jump_idx_list[interval_idx]+1)

        found_meas = np.array(meas_at_pos)[meas_idx_range]

        return found_meas

    def get_meas_from_filenames(self):
        sam_paths = self.sample_paths.selected_paths
        cache_map = self.cache.filepath_map
        sam_meas_list = [cache_map[p] for p in sam_paths if p.is_file() and p in cache_map]

        return sam_meas_list

    def get_selected_measurements(self):
        identifier_map = {
            SelectionCriterionEnum.selected_timestamp: self.sel_timestamp,
            SelectionCriterionEnum.selected_point: self.sel_point,
            SelectionCriterionEnum.string_search: self.string_match,
            SelectionCriterionEnum.file_selection: None,
        }
        handlers = {
            SelectionCriterionEnum.selected_timestamp: self.get_measurements_from_timestamp,
            SelectionCriterionEnum.selected_point: lambda pnt: self.get_measurements_from_point(*pnt),
            SelectionCriterionEnum.string_search: self.get_measurements_from_string,
            SelectionCriterionEnum.file_selection: self.get_meas_from_filenames,
        }

        handler = handlers[self.selection_criterion]
        identifier = identifier_map[self.selection_criterion]
        selected_meas = handler() if identifier is None else handler(identifier)

        if not selected_meas:
            return []

        if len(selected_meas) == 1:
            s = f"{selected_meas[0].filepath.name}"
        else:
            s = f"{selected_meas[0].filepath.name} -\n{selected_meas[-1].filepath.name}"

        self.dataset.info_pane.set_trait("selected_measurement_info", s)

        return selected_meas

    def get_arb_line(self, p0, p1):
        # Bresenham's_line_algorithm
        scale_x = 1 / self.dataset.shape_properties["dx"]
        scale_y = 1 / self.dataset.shape_properties["dy"]

        p0 = p0.magnitude if isinstance(p0, Q_) else p0
        p1 = p1.magnitude if isinstance(p1, Q_) else p1

        x0, y0 = round(p0[0] * scale_x), round(p0[1] * scale_y)
        x1, y1 = round(p1[0] * scale_x), round(p1[1] * scale_y)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy
        curr_x, curr_y = x0, y0
        points = []

        while True:
            points.append((curr_x / scale_x, curr_y / scale_y))

            if curr_x == x1 and curr_y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

        meas_list = meas_list = [meas for p in points for meas in self.get_measurements_from_point(*p)]
        meas_list = list(dict.fromkeys(meas_list))
        points = [meas.position for meas in meas_list]

        return meas_list, points

    def get_nearest_ref(self, meas_, dist_func=None, meas_set=None):
        if not dist_func:
            dist_func = self.dist_func.value
        if meas_set is None:
            meas_set = self.measurements["refs"]

        closest_ref, best_fit_val = None, np.inf
        for ref_meas in meas_set:
            dist_val = dist_func(ref_meas, meas_)
            if np.abs(dist_val) < np.abs(best_fit_val):
                best_fit_val = dist_val
                closest_ref = ref_meas
        # from random import choice
        # closest_ref = choice(self.measurements["refs"])

        self.dataset.logger.debug(f"Sam: {meas_})")
        self.dataset.logger.debug(f"Ref: {closest_ref})")
        if self.dist_func == Dist.Time:
            self.dataset.logger.debug(f"Time between ref and sample: {best_fit_val} seconds")
        else:
            self.dataset.logger.debug(f"Distance between ref and sample: {best_fit_val} mm")
        if closest_ref is None:
            self.dataset.logger.warning("No nearest reference found, returning first reference")
            return self.measurements["refs"][0]

        return closest_ref

    def ref_file_selection_to_ref_meas(self):
        if self.ref_sel_criterion != ReferenceSelection.file_selection:
            return []
        ref_list = [self.cache.filepath_map[p] for p in self.reference_paths.selected_paths if p.is_file()]
        return ref_list

    def _ref_file_selection(self, meas_list):
        ref_list = self.ref_file_selection_to_ref_meas()
        rl_len, ml_len = len(ref_list), len(meas_list)
        if rl_len != ml_len:
            self.set_trait("direct_match", False)
        else:
            self.set_trait("direct_match", True)
        if ml_len == 0:
            return ref_list
        if rl_len == 0:
            ref_list = ml_len * [self.measurements["max_amp_meas"]]
            logging.info("No reference files selected, using maximum amplitude measurement")
        elif rl_len != ml_len:
            if rl_len < ml_len:
                ref_list.extend((ml_len - rl_len) * [ref_list[-1]])
            elif rl_len > ml_len:
                ref_list = ref_list[:ml_len]

        ref_list = [self.get_nearest_ref(meas, meas_set=ref_list) for meas in meas_list]

        return ref_list

    def get_matching_refs(self, meas_list):
        if len(self.measurements["refs"]) == 0:
            return []

        ref_getter = None
        match self.ref_sel_criterion:
            case ReferenceSelection.file_selection:
                return self._ref_file_selection(meas_list)
            case ReferenceSelection.closest_distance:
                ref_getter = lambda meas: self.get_nearest_ref(meas)
                self.dataset.logger.debug(f"Using reference measurement closest to {self.dataset.ref_point} as ref.")
            case ReferenceSelection.max_amp_measurement:
                ref_getter = lambda meas: self.measurements["max_amp_meas"]
                self.dataset.logger.debug("Using the measurement with the highest amplitude as reference")
            case ReferenceSelection.fix_ref:
                ref_idx = min(len(self.measurements["refs"]) - 1, self.fix_ref_idx)
                ref_getter = lambda meas: self.measurements["refs"][ref_idx]

        return [ref_getter(meas) for meas in meas_list] if ref_getter else []

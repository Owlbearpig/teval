import numpy as np
from common.components import ComponentBase
from common.traits import MultiPathSelection, ValueRange, MultiPathClass
from common.units import Q_
from common.default_appsettings import Dist
from traitlets import Enum as TEnum, Unicode, Bool, Int
from enum import Enum
from common.measurements import timestamp2id

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
    max_amp_measurement = "Maximum amplitude measurement"
    closest_distance = "Closest distance"
    fix_ref = "Use fixed index reference"
    file_selection = "File selection"

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
                              ReferenceSelection.max_amp_measurement).tag(name="Reference matching criterion",
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
                        ).tag(name="Direct file match", priority=2000, group=reference_matching_grp)

    reference_paths = MultiPathSelection().tag(fullwidth = False, group="Direct reference file selection", combine=True)
    sample_paths = MultiPathSelection().tag(fullwidth = False, group="Direct sample file selection", combine=True)


    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)

        self.dataset = dataset

        ref_filenames = [f"{meas.filepath.name}" for meas in self.dataset.measurements["refs"]]
        sam_filenames = [f"{meas.filepath.name}" for meas in self.dataset.measurements["sams"]]

        self.reference_paths = MultiPathClass(root_path=self.dataset.data_path, shown_filenames=ref_filenames)
        self.sample_paths = MultiPathClass(root_path=self.dataset.data_path, shown_filenames=sam_filenames)

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

        self.reference_paths = MultiPathClass(root_path=root_path, shown_filenames=ref_filenames)
        self.sample_paths = MultiPathClass(root_path=root_path, shown_filenames=sam_filenames)

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

    def update_sel_cnt_info(self, change):
        change_name = change["name"]
        if change_name in ["selected_ref_cnt", "selected_sam_cnt"]:
            return
        selected_measurements = self.selected_measurements
        if selected_measurements is None:
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

        sam_meas_list = [self.cache.filepath_map[p] for p in sam_paths if p.is_file()]

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

    def _ref_file_selection(self, meas_list):
        ref_list = [self.cache.filepath_map[p] for p in self.reference_paths.selected_paths if p.is_file()]
        rl_len, ml_len = len(ref_list), len(meas_list)
        if rl_len != ml_len:
            self.set_trait("direct_match", False)
        else:
            self.set_trait("direct_match", True)
        if rl_len == 0:
            ref_list = ml_len * [self.measurements["max_amp_meas"]]
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

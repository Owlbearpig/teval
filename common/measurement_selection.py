from logging import log

from common.components import ComponentBase
from common.traits import MultiPathSelection, ValueRange, MultiPathClass
from common.units import Q_
from traitlets import Enum as TEnum, Unicode, Bool
from enum import Enum
from common.measurements import timestamp2id

class SelectionCriterionEnum(Enum):
    file_selection = "File selection"
    selected_timestamp = "Timestamp"
    selected_point = "Point"
    string_search = "String"

class ReferenceSelection(Enum):
    point_as_ref = "Single point"
    max_amp_measurement = "Maximum amplitude measurement"
    closest_distance = "Closest distance"
    fix_ref = "Use fixed index reference"
    file_selection = "File selection"

class MeasurementSelection(ComponentBase):
    average_selection = Bool(False).tag(name="Allow averaging")


    measurement_selection_grp = "Measurement selection"
    selection_criterion = TEnum(SelectionCriterionEnum,
                                SelectionCriterionEnum.file_selection).tag(name="Select measurement by",
                                                                           group=measurement_selection_grp,
                                                                           priority=-2)
    sel_point = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(name="Selected point (x, y)",
                                                                             group=measurement_selection_grp)
    sel_timestamp = Unicode("").tag(name="Selected timestamp", group=measurement_selection_grp)
    string_match = Unicode("").tag(name="Filter string", group=measurement_selection_grp)

    reference_selection_grp = "Reference selection"
    ref_sel_criterion = TEnum(ReferenceSelection,
                              ReferenceSelection.max_amp_measurement).tag(name="Reference selection criterion",
                                                                     group=reference_selection_grp,
                                                                     priority=-2)
    ref_pos = ValueRange([0, 0], group=reference_filter_group).tag(name="Reference position")
    dist_func = TEnum(Dist, default_value=Dist.Time).tag(priority=1000, name="Measurement distance function",
                                                         group=reference_selection_grp)
    fix_ref_idx = Int(0, min=-1, group=reference_selection_grp).tag(name="Fixed reference index")

    references = MultiPathSelection().tag(fullwidth = False, group="Direct reference selection", combine=True)
    samples = MultiPathSelection().tag(fullwidth = False, group="Direct sample selection", combine=True)


    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)

        self.dataset = dataset

        self.references = MultiPathClass(root_path=self.dataset.data_path)
        self.samples = MultiPathClass(root_path=self.dataset.data_path)

        self.dataset.observe(self.update, "data_path")

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
    def reference_measurements(self):
        return self.get_matching_ref(self.selected_measurements)

    def update(self, change):
        new_path = change["new"]
        self.references = MultiPathClass(root_path=new_path)
        self.samples = MultiPathClass(root_path=new_path)

    def get_measurements_from_point(self, x, y):
        if self.cache is None:
            logging.info("Cache not loaded, check dataset path")
            return None
        meas_list = self.measurements["all"]
        if isinstance(x, Q_):
            x = x.magnitude
        if isinstance(y, Q_):
            y = y.magnitude
        pnt = (x, y)

        key = self.cache.coord_map_key_func(pnt)
        found_meas_list = self.cache.coord_map[key]

        return found_meas_list

    def get_measurements_from_timestamp(self, timestamp_str=""):
        if not timestamp_str:
            logging.warning("No timestamp set. Returning first measurement")
            return self.measurements["all"][0]
        logging.info("Selecting measurement by timestamp", timestamp_str)
        meas_id_ = timestamp2id(timestamp_str)

        found_meas_list = []
        for meas in self.measurements["all"]:
            if meas.identifier == meas_id_:
                found_meas_list.append(meas)

        if not found_meas_list:
            logging.warning(f"No measurement with timestamp: {timestamp_str} (id: {meas_id_}) found in dataset")

        return found_meas_list

    def get_measurements_from_string(self, string=""):
        if not string:
            logging.warning("No filter string set. Returning first measurement")
            return self.measurements["all"][0]

        found_meas_list = []
        for meas in self.measurements["all"]:
            if string in meas.filepath.name:
                found_meas_list.append(meas)

        if not found_meas_list:
            logging.warning(f"No measurement containing the string {string} in the filename found in dataset")

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
        sam_paths = self.samples.selected_paths

        sam_meas_list = [self.cache.filepath_map[p] for p in sam_paths]

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

        meas_list = [self.get_measurements_from_point(*p) for p in points]
        meas_list = list(dict.fromkeys(meas_list))
        points = [meas.position for meas in meas_list]

        return meas_list, points

    def get_cart_line(self, x=None, y=None, limits=None):
        shape = self.dataset.shape_properties
        if x is None and y is None:
            return None

        x_coords, y_coords = shape["x_coords"], shape["y_coords"]

        # vertical direction / slice
        if x is not None:
            ret = [self.get_measurements_from_point(x, y_) for y_ in y_coords], y_coords
        else:  # horizontal direction / slice
            ret = [self.get_measurements_from_point(x_, y) for x_ in x_coords], x_coords

        if limits is None:
            return ret
        else:
            measurements, coords = ret
            meas_in_limit_range = []
            for i, coord in enumerate(coords):
                if (limits[0] < coord) and (coord < limits[1]):
                    meas_in_limit_range.append(measurements[i])

            return meas_in_limit_range, coords

    def get_nearest_ref(self, meas_, dist_func=None, excluded_refs=None):
        if not dist_func:
            dist_func = self.dist_func.value
        if not excluded_refs:
            excluded_refs = []
        closest_ref, best_fit_val = None, np.inf
        for ref_meas in self.measurements["refs"]:
            if ref_meas in excluded_refs:
                continue
            dist_val = dist_func(ref_meas, meas_)
            if np.abs(dist_val) < np.abs(best_fit_val):
                best_fit_val = dist_val
                closest_ref = ref_meas
        # from random import choice
        # closest_ref = choice(self.measurements["refs"])

        logging.debug(f"Sam: {meas_})")
        logging.debug(f"Ref: {closest_ref})")
        if self.dist_func == Dist.Time:
            logging.debug(f"Time between ref and sample: {best_fit_val} seconds")
        else:
            logging.debug(f"Distance between ref and sample: {best_fit_val} mm")
        if closest_ref is None:
            logging.warning("No nearest reference found, returning first reference")
            return self.measurements["refs"][0]

        return closest_ref

    def get_matching_ref(self, meas_list):
        ref_getter = None
        match self.ref_sel_criterion:
            case ReferenceSelection.file_selection:
                ref_list = [self.cache.filepath_map[p] for p in self.references.selected_paths]
                if len(ref_list) != len(meas_list):
                    loggging.warning("Reference file selection length != number of selected measurements")
                    if len(ref_list) < len(meas_list):
                        ref_list.extend((len(meas_list)-len(ref_list))*[ref_list[-1]])
                    elif len(ref_list) > len(meas_list):
                        ref_list = ref_list[:len(meas_list)]

                return ref_list
            case ReferenceSelection.closest_distance:
                ref_getter = lambda meas: self.get_nearest_ref(meas)
            case ReferenceSelection.point_as_ref:
                ref_getter = lambda meas: self.get_measurements_from_point(*self.ref_pos)
                logging.info(f"Using measurement closest to {self.ref_pos} as ref.")
            case ReferenceSelection.max_amp_measurement:
                ref_getter = lambda meas: self.measurements["max_amp_meas"]
                logging.info("Using the measurement with the highest amplitude as reference")
            case ReferenceSelection.fix_ref:
                ref_getter = lambda meas: self.measurements["refs"][self.fix_ref_idx]

        return [ref_getter(meas) for meas in meas_list] if ref_getter else []

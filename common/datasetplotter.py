from common.components import ComponentBase, action
from common.dataset import DataSet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from common.default_appsettings import Domain, Dist, Direction, ClimateQuantity, QuantityFunc, QuantityEnum
from functools import partial
from common.functions import moving_average, f_axis_idx_map, local_minima_1d, round_dx
import logging
from datetime import datetime
from common.measurements import Measurement
from scipy.special import erfc
from scipy.optimize import curve_fit
from common.eval_component.shgo import shgo
from traitlets import Float, observe, Bool, Unicode, Enum as TEnum
from common.traits import Q_, Quantity, ValueRange
from mpl_settings import mpl_style_params
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
from matplotlib.backend_bases import MouseButton


action = partial(action, check_init=True, rc_params=mpl_style_params)

class DataSetPlotter(ComponentBase):

    sel_freq_range = ValueRange(default_value=[Q_(1.000, "THz"), Q_(1.200, "THz")]).tag(
        name="Selected frequency range", priority=1000,
    )
    comparison_point = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(name="Point for comparison (x, y)")

    selected_quantity = TEnum(QuantityEnum, default_value=QuantityEnum.P2P).tag(name="Selected quantity", priority=1001)
    quantity_value = Unicode("", read_only=True).tag(name="Quantity value", priority=1003)

    rect_sel_grp = "Average value rectangle"
    rect_sel_label = Unicode("").tag(name="Rectangle label", priority=1000, group=rect_sel_grp)
    rect_sel_bot_left = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(
        name="Bottom left rectangle selection", group=rect_sel_grp, priority=1001)
    rect_sel_top_right = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(
        name="Top right rectangle selection", group=rect_sel_grp, priority=1002)

    image_grp = "Image actions"
    confine_to_extent = Bool(False,
                             group=image_grp).tag(name="Confine image to extent", priority=1001)
    img_extent_x_range = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(
        name="Set image x-range", group=image_grp, priority=1002)
    img_extent_y_range = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(
        name="Set image y-range", group=image_grp, priority=1003)
    enable_img_interaction = Bool(True, group=image_grp).tag(name="Enable interaction")

    line_plot_grp = "Image slice plot"
    line_start = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(
        name="Line start point", group=line_plot_grp, priority=1001)
    line_end = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(
        name="Line end point", group=line_plot_grp, priority=1002)

    def __init__(self, dataset : DataSet, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

        self.grid_vals = None
        self.img_ax = None
        self.drawn_elements = {"patches": [], "text_labels": [], "points": []}

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.settings.save_configuration(self)

    def __enter__(self):
        self.settings.load_configuration(self)

        return self

    @property
    def freq_idx(self, freq=None):
        if freq is None:
            freq = self.sel_freq_range[0]
        selected_freq_idx = f_axis_idx_map(self.dataset.freq_axis, freq)

        return selected_freq_idx

    @property
    def plot_freq_axis(self):
        return self.dataset.freq_axis[self.plot_idx_range]

    @property
    def plot_idx_range(self):
        return f_axis_idx_map(self.dataset.freq_axis, self.plot_settings.plot_range)

    @property
    def settings(self):
        return self.dataset.settings

    @property
    def plot_settings(self):
        return self.dataset.settings.plot_opt

    @property
    def measurements(self):
        return self.dataset.measurements

    @property
    def img_shape(self):
        return self.dataset.shape_properties

    @property
    def scalar_func(self):
        sel_quant = self.get_selected_quantity()
        def grid_func(meas, func=sel_quant.func):
            sel_freq_idx = self.freq_idx

            res = np.real(func(meas))
            ndim = np.ndim(res)
            if ndim == 0:
                val = res
            elif ndim == 1:
                val = res[sel_freq_idx[0]]
            else:
                val = res[sel_freq_idx[0], 1]

            return val

        return grid_func

    @property
    def td_fig_num(self):
        fig_num_ext = self.plot_settings.fig_num_ext
        return "Time domain" + fig_num_ext

    @property
    def fd_fig_num(self):
        fig_num_ext = self.plot_settings.fig_num_ext
        return "Frequency domain" + fig_num_ext

    @property
    def image_fig_num(self):
        sel_quant = self.get_selected_quantity()
        en_freq_label = Domain.Frequency == sel_quant.domain
        fig_num = ""
        if self.plot_settings.img_fig_num_ext:
            fig_num += self.plot_settings.img_fig_num_ext + " "
        fig_num += str(sel_quant)

        f1, f2 = int(self.sel_freq_range[0].magnitude * 1e3), int(self.sel_freq_range[1].magnitude * 1e3)
        if np.isclose(f1, f2):
            fig_num += en_freq_label * f" {f1} GHz"
        else:
            fig_num += en_freq_label * f" {f1}-{f2} GHz"
        fig_num = fig_num.replace(" ", "_")

        return fig_num

    @property
    def quantity_label(self):
        sel_quant = self.get_selected_quantity()
        en_freq_label = Domain.Frequency == sel_quant.domain
        if np.isclose(self.sel_freq_range[0].magnitude, self.sel_freq_range[1].magnitude):
            freq_label = f"({self.sel_freq_range[0]})"
        else:
            freq_label = f"({self.sel_freq_range[0]}-{self.sel_freq_range[1]})"

        return " ".join([str(sel_quant), freq_label * en_freq_label])

    def get_selected_measurement(self, **kwargs):
        return self.dataset.get_selected_measurement(**kwargs)

    def get_selected_quantity(self):
        self._update_quant_func()
        return self.selected_quantity.value

    def _get_empty_grid(self):
        img_shape = self.img_shape
        w, h = img_shape["w"], img_shape["h"]
        grid_vals = np.zeros((w, h), dtype=complex)

        return grid_vals

    def _calc_grid_vals(self):
        grid = self._get_empty_grid()
        sam_meas = self.measurements["sams"]

        iter_ = tqdm(enumerate(sam_meas), total=len(sam_meas),
                     desc="Evaluating measurements", colour="green")
        for i, measurement in iter_:
            x_idx, y_idx = self._coords_to_idx(*measurement.position)

            grid[x_idx, y_idx] = self.scalar_func(measurement)

        return grid

    def _update_quant_func(self):
        func_map = self.dataset.func_map

        freq_range = (self.sel_freq_range[0].magnitude, self.sel_freq_range[1].magnitude)
        func_map[QuantityEnum.PowerInt] = partial(self.dataset.power_int, freq_range=freq_range)
        func_map[QuantityEnum.PeakCnt] = partial(self.dataset.simple_peak_cnt, threshold=2.5)

        self.selected_quantity.value.func = func_map[self.selected_quantity]

    @observe("selected_quantity")
    def select_quantity(self, change=None):
        if change is None:
            return
        self._update_quant_func()

    def _coords_to_idx(self, x_, y_):
        shape_properties = self.img_shape
        x, y = shape_properties["x_coords"], shape_properties["y_coords"]
        x_idx, y_idx = np.argmin(np.abs(x_ - x)), np.argmin(np.abs(y_ - y))

        return x_idx, y_idx

    def _idx_to_coords(self, x_idx, y_idx):
        shape_properties = self.img_shape
        dx, dy = shape_properties["dx"], shape_properties["dy"]

        y = shape_properties["y_coords"][0] + y_idx * dy
        x = shape_properties["x_coords"][0] + x_idx * dx

        return x, y

    def _is_excluded(self, idx_tuple):
        excl_areas = None # not implemented
        if excl_areas is None:
            return False

        if np.array(excl_areas).ndim == 1:
            areas = [excl_areas]
        else:
            areas = excl_areas

        for area in areas:
            x, y = self._idx_to_coords(*idx_tuple)
            return (area[0] <= x <= area[1]) * (area[2] <= y <= area[3])

        return False

    def _exclude_pixels(self, grid_vals):
        empty_grid = self._get_empty_grid()
        filtered_grid = grid_vals.copy()
        dims = filtered_grid.shape
        for x_idx in range(dims[0]):
            for y_idx in range(dims[1]):
                if self._is_excluded((x_idx, y_idx)):
                    filtered_grid[x_idx, y_idx] = empty_grid[x_idx, y_idx]

        return filtered_grid

    def get_meas_quantities(self, only_ret_quants=False):
        ref_meas, selected_meas = self.get_selected_measurement()
        meas_quants = self.dataset.calc_meas_quantities(ref_meas, selected_meas)

        logging.info(f"Reference measurement: {ref_meas}")
        logging.info(f"Sample measurement: {selected_meas}\n")

        if self.plot_settings.shift_sam2ref:
            t0_sam = np.argmax(meas_quants["sam_td"][:, 1])
            t0_ref = np.argmax(meas_quants["ref_td"][:, 1])
            shift_t = np.abs(t0_ref - t0_sam)
            meas_quants["sam_td"][:, 1] = np.roll(meas_quants["sam_td"][:, 1], -shift_t)

        if self.plot_settings.remove_t_offset:
            meas_quants["sam_td"][:, 0] -= meas_quants["sam_td"][0, 0]

        if only_ret_quants:
            return meas_quants

        return ref_meas, selected_meas, meas_quants

    @action("Calculate quantity value", priority=1002)
    def calc_quant_value(self):
        selected_meas = self.get_selected_measurement(also_return_ref=False)
        sel_quant = self.get_selected_quantity()
        value = sel_quant.func(selected_meas)

        if not isinstance(value, np.ndarray):
            unit = sel_quant.unit
            s = f"{value:.2f} {unit}" if unit else f"{value:.2f}"
            self.set_trait("quantity_value", s)
        else:
            logging.info("Selected quantity is not a scalar")

    def set_time_axis_unit(self, axis):
        axis = axis.to("h")
        if axis.magnitude.max() < 5 / 60:
            axis = axis.to("s")
        elif 5 / 60 <= axis.magnitude.max() < 0.5:
            axis = axis.to("min")
        else:
            axis = axis.to("h")

        return axis

    def get_stability_data(self, meas_set_kw=None):
        if meas_set_kw is not None:
            meas_set = []
            for meas in self.measurements["all"]:
                if meas_set_kw in meas.filepath.name:
                    meas_set.append(meas)
            logging.info(f"Using measurements containing keyword {meas_set_kw}")
        elif all([self.measurements["all"][0].position == meas.position for meas in self.measurements["all"]]):
            meas_set = self.measurements["all"]
            logging.info("Using all measurements")
        else:
            meas_set = self.measurements["refs"]
            logging.info("Using reference measurement set")
            if len(meas_set) < 2:
                msg = "Not enough measurements assigned as reference in dataset. "
                msg += "Using all measurements instead"
                logging.info(msg)
                meas_set = self.measurements["all"]

        if len(meas_set) == 0:
            logging.info("No measurements in selected measurement set")
            return None

        meas0 = meas_set[0]
        n_meas = len(meas_set)
        ret = {
            "meas_set": meas_set,
            "meas_times": Q_(np.zeros(n_meas), "h"),
            "ampl_arr": np.zeros(n_meas),
            "zero_crossing": np.zeros(n_meas),
            "relative_delay": np.zeros(n_meas),
            "angle_arr": np.zeros(n_meas),
            "spec_similarity": np.zeros(n_meas),
        }

        for i, meas_ in enumerate(meas_set):
            ref_td, ref_fd = self.dataset.get_data(meas_, domain=Domain.Both)
            fd_val = ref_fd[self.freq_idx, 1]

            ret["meas_times"][i] = self.dataset.meas_time_diff(meas0, meas_)
            ret["zero_crossing"][i] = self.dataset.get_zero_crossing(meas_)
            ret["relative_delay"][i] = self.dataset.delay_from_phaseslope(meas0, meas_)
            ret["ampl_arr"][i] = np.abs(fd_val)
            ret["angle_arr"][i] = -np.angle(fd_val)
            ret["spec_similarity"][i] = self.dataset.spectral_similarity(meas0, meas_)

        ret["meas_times"] = self.set_time_axis_unit(ret["meas_times"])

        return ret

    @action(name="Draw average value rectangle", group=rect_sel_grp, priority=2)
    def average_area(self):
        pnt_bot_left = self.rect_sel_bot_left.magnitude
        pnt_top_right = self.rect_sel_top_right.magnitude
        assert (pnt_bot_left[0] <= pnt_top_right[0]) and (pnt_bot_left[1] <= pnt_top_right[1])

        x_coords, y_coords = self.img_shape["x_coords"], self.img_shape["y_coords"]
        pnt_bot_left = (x_coords[np.argmin(np.abs(pnt_bot_left[0] - x_coords))],
                        y_coords[np.argmin(np.abs(pnt_bot_left[1] - y_coords))])
        pnt_top_right = (x_coords[np.argmin(np.abs(pnt_top_right[0] - x_coords))],
                         y_coords[np.argmin(np.abs(pnt_top_right[1] - y_coords))])

        x0_idx, y0_idx = self._coords_to_idx(*pnt_bot_left)
        x1_idx, y1_idx = self._coords_to_idx(*pnt_top_right)

        if self.grid_vals is None:
            self._calc_grid_vals()

        grid_vals = self.grid_vals[x0_idx:x1_idx+1, y0_idx:y1_idx+1]
        mean_val, std_val = np.mean(grid_vals), np.std(grid_vals)

        mean_s, std_s = str(np.round(mean_val, 4)), str(np.round(std_val, 4))
        min_s, max_s = str(np.round(np.min(grid_vals), 4)), str(np.round(np.max(grid_vals), 4))

        label = self.rect_sel_label
        meas_cnt = grid_vals.shape[0] * grid_vals.shape[1]
        quant = self.selected_quantity
        logging.info(f"Average {quant.name} value of area {label} containing {meas_cnt} measurements: {mean_s}±{std_s}")
        logging.info(f"Min: {min_s}, max: {max_s}\n")

        ret = {"mean": mean_val, "std": std_val, "min": min_s, "max": max_s}
        if not plt.fignum_exists(num=self.image_fig_num):
            return ret

        plt.figure(self.image_fig_num)
        ax = self.img_ax

        dx, dy = self.img_shape["dx"], self.img_shape["dy"]
        # pixels are centered around each coordinate, unlike patches.Rectangle
        pnt_bot_left = (pnt_bot_left[0] - dx / 2, pnt_bot_left[1] - dy / 2)
        pnt_top_right = (pnt_top_right[0] + dx / 2, pnt_top_right[1] + dy / 2)

        # draw rectangle
        rect_width = pnt_top_right[0] - pnt_bot_left[0]
        rect_height = pnt_top_right[1] - pnt_bot_left[1]
        rect = patches.Rectangle(
            pnt_bot_left,  # bottom left
            rect_width,  # width
            rect_height,  # height
            linewidth=2, edgecolor="black", facecolor="none"
        )
        patch = ax.add_patch(rect)

        # decide where to put rect label
        img_extent = self.img_shape["extent"]
        t_x, t_y = pnt_bot_left[0] + rect_width / 2, pnt_bot_left[1] + rect_height / 2
        if t_x < img_extent[0] + 1:
            t_x = pnt_bot_left[0] + rect_width + 1.5
        if t_x > img_extent[1] - 1:
            t_x = pnt_top_right[0] - rect_width - 1.5

        t_y_below = pnt_top_right[1] - rect_height - 1.5
        t_y_above = pnt_bot_left[1] + rect_height + 1.5
        if t_y < img_extent[2] + 1:
            t_y = t_y_above
        if t_y > img_extent[3] - 1:
            t_y = t_y_below

        # if rect too small, place above or below
        if rect_height < 3.0 or rect_width < 3.0:
            # if in top half, place below, else place above
            if t_y > img_extent[2] + 0.5 * abs(img_extent[3] - img_extent[2]):
                t_y = t_y_below
            else:
                t_y = t_y_above

        # add label
        text_label = ax.text(t_x, t_y, label,
                             color="black", fontsize=18, ha="center", va="center", fontweight="bold")

        plt.draw()

        self.drawn_elements["patches"].append(patch)
        self.drawn_elements["text_labels"].append(text_label)

        return ret

    @action("Clear drawn elements", group=image_grp, priority=3)
    def clear_rectangles(self):
        for patch in self.drawn_elements["patches"]:
            patch.remove()
        for text_label in self.drawn_elements["text_labels"]:
            text_label.remove()
        for point in self.drawn_elements["points"]:
            point.remove()

        self.drawn_elements["patches"].clear()
        self.drawn_elements["text_labels"].clear()
        self.drawn_elements["points"].clear()

        if plt.fignum_exists(num=self.image_fig_num):
            plt.draw()

    @action("Reference difference", group="Plots")
    def ref_difference_plot(self):
        if len(self.measurements["refs"]) < 2:
            logging.warning("Cannot plot reference difference with less than 2 reference measurements")
            return
        meas_idx = np.random.randint(len(self.measurements["refs"])-1)
        ref1, ref2 = self.measurements["refs"][meas_idx], self.measurements["refs"][meas_idx+1]
        logging.info(f"Reference difference:")
        logging.info(f"First measurement: {ref1}")
        logging.info(f"Second measurement: {ref2}")
        # print(ref1)
        # print(ref2)

        ref1_fd = self.dataset.get_data(ref1, domain=Domain.Frequency)
        ref2_fd = self.dataset.get_data(ref2, domain=Domain.Frequency)

        freq = ref1_fd[:, 0]

        phi1 = np.angle(ref1_fd[:, 1])
        phi2 = np.angle(ref2_fd[:, 1])

        phi1_unwrap = np.unwrap(phi1)
        phi2_unwrap = np.unwrap(phi2)

        diff = phi1_unwrap - phi2_unwrap
        diff_diff = np.append(np.diff(diff), 0)

        plt.figure("Unwrapped phases")
        plt.plot(freq, phi1_unwrap, label="Ref 1")
        plt.plot(freq, phi2_unwrap, label="Ref 2")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Unwrapped phase (rad)")

        plt.figure("Phase difference")
        plt.scatter(freq, diff_diff)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase difference (rad)")

        ref0 = self.measurements["refs"][0]
        dt1 = self.dataset.meas_time_diff(ref0, ref1)
        dt2 = self.dataset.meas_time_diff(ref0, ref2)

        mark_x = [dt1.magnitude, dt2.magnitude]
        mark_y = [phi1[self.freq_idx], phi2[self.freq_idx]]

        plt.figure("Stability phase")
        plt.scatter(mark_x, mark_y, color="red", s=30, zorder=99)
        plt.xlabel(f"Time since first measurement ({dt1.units})")
        plt.ylabel("Phase (rad)")

        # self.plt_show()

    @action("Reference measurement", group="Plots")
    def plot_ref(self, ref_meas_=None, timestamp=None, ref_idx=None):
        label = None

        if (ref_meas_ is None) and (timestamp is None):
            if isinstance(ref_idx, int):
                ref_meas_ = self.measurements["refs"][ref_idx]
                label = f"Reference idx: {ref_idx}"
            else:
                ref_meas_ = self.measurements["refs"][0]
                label = "Reference idx: 0"
        elif ref_meas_ is None:
            ref_meas_ = self.dataset.get_measurement_from_timestamp(timestamp)
            if ref_meas_ is None:
                return

        zero_crossing = self.dataset.get_zero_crossing(ref_meas_)
        # print(zero_crossing)
        # zx_simple = self._get_zero_crossing(ref_meas_) - self._get_zero_crossing(self.measurements["refs"][0])
        # zx_phase = self._delay_from_phaseslope(self.measurements["refs"][0], ref_meas_)
        # print(zx_simple*1e3, zx_phase*1e3)

        ref_td, ref_fd = self.dataset.get_data(ref_meas_, domain=Domain.Both)
        freq_axis = ref_fd[:, 0].real
        plot_range = self.plot_settings.plot_range
        f_idx_range = f_axis_idx_map(self.dataset.freq_axis, plot_range)

        if self.plot_settings.remove_t_offset:
            ref_td[:, 0] -= ref_td[0, 0]

        label = label if label else f"Reference ({ref_meas_.meas_time})"

        sub_noise_floor = self.plot_settings.sub_noise_floor
        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        y_db = (20 * np.log10(np.abs(ref_fd[f_idx_range, 1])) - noise_floor).real
        plt.figure(self.fd_fig_num)
        plt.plot(freq_axis[f_idx_range], y_db, label=label + " (Reference)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.draw()

        plt.figure(self.td_fig_num)
        plt.plot(ref_td[:, 0], ref_td[:, 1], label=label + " (Reference)")
        if self.plot_settings.plot_zero_crossing:
            plt.scatter(zero_crossing, 0, label="Zero-crossing", color="red")
        # plt.plot(ref_td[1:, 0], np.diff(np.abs(ref_td[:, 1])), label=label)
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (Arb. u.)")
        plt.draw()

    @action("Waveform", group="Plots")
    def plot_waveform(self, point=None):
        if point is not None:
            sam_meas = self.dataset.get_measurement_from_point(*point)
            ref_meas = self.dataset.get_nearest_ref(sam_meas)
            meas_quants = self.dataset.calc_meas_quantities(ref_meas, sam_meas)
        else:
            ref_meas, sam_meas, meas_quants = self.get_meas_quantities()
            point = sam_meas.position

        if not plt.fignum_exists(self.td_fig_num) or not plt.fignum_exists(self.fd_fig_num):
            self.plot_ref(ref_meas)

        sam_td, sam_fd = meas_quants["sam_td"], meas_quants["sam_fd"]

        sub_noise_floor = self.plot_settings.sub_noise_floor
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        label = self.plot_settings.label
        if not label:
            label = f"(x,y)=({point[0]}, {point[1]})"

        freq_axis = self.dataset.freq_axis
        f_idx_range = f_axis_idx_map(freq_axis, self.plot_settings.plot_range)

        td_scale = self.plot_settings.td_scale

        plt.figure(self.td_fig_num)
        td_label = label
        if not np.isclose(td_scale, 1):
            td_label += f"\n(Amplitude x {td_scale})"
        plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=td_label)
        plt.draw()

        plt.figure(self.fd_fig_num)
        y_db = (20 * np.log10(np.abs(sam_fd[f_idx_range, 1])) - noise_floor).real
        plt.plot(freq_axis[f_idx_range], y_db, label=label)
        plt.draw()

        return sam_meas

    @action("Phase plots", group="Phase plots")
    def plot_phase_plot(self):
        selected_meas = self.get_selected_measurement(also_return_ref=False)

        simple_eval_res = self.dataset.windowing_eval(selected_meas)
        phi = simple_eval_res["phi"][self.plot_idx_range]
        phi_corrected = simple_eval_res["phi_corrected"][self.plot_idx_range]

        fig_num_ext = self.plot_settings.fig_num_ext
        label = self.plot_settings.label
        if not label:
            label = str(selected_meas.filepath.name)

        plt.figure("Phase correction comparison" + fig_num_ext)
        plt.plot(self.plot_freq_axis, phi, label=label + " (Original)", ls="dashed")
        plt.plot(self.plot_freq_axis, phi_corrected, label=label + " (Corrected)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")

        plt.figure("Phase" + fig_num_ext)
        plt.plot(self.plot_freq_axis, phi_corrected, label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")

        plt.figure("Phase slope" + fig_num_ext)
        plt.plot(self.plot_freq_axis[:-1], np.diff(phi_corrected), label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad/THz)")

    @action("Plot selected quantity", group="Plots")
    def plot_selected_quantity(self, point=None):
        fig_num_ext = self.plot_settings.fig_num_ext
        if point is None:
            ref_meas, selected_meas = self.get_selected_measurement()
            point = selected_meas.position
        else:
            selected_meas = self.dataset.get_measurement_from_point(*point)
            ref_meas = self.dataset.get_nearest_ref(selected_meas)

        sel_quant = self.get_selected_quantity()
        values = sel_quant.func(selected_meas)
        label = self.plot_settings.label
        if not label:
            label = f"(x,y)=({point[0]}, {point[1]})"

        if not isinstance(values, np.ndarray):
            logging.info("Selected quantity is a scalar")
            return

        values = values[self.plot_idx_range]

        is_single_plot = True if not np.issubdtype(values.dtype, np.complexfloating) else False
        fignum = str(sel_quant) + fig_num_ext
        y_label = f"{sel_quant} ({sel_quant.unit})" if sel_quant.unit else f"{sel_quant}"
        if is_single_plot:
            plt.figure(fignum)
            plt.plot(self.plot_freq_axis, values, label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel(y_label)
        else:
            if not plt.fignum_exists(fignum):
                fig, (ax0, ax1) = plt.subplots(2, 1, num=fignum, sharex=True, gridspec_kw={'hspace': 0})
                ax1.set_xlabel("Frequency (THz)")
                ax0.set_ylabel(f"{y_label} (Real part)")
                ax1.set_ylabel(f"{y_label} (Imag part)")
            else:
                fig = plt.figure(fignum)
                ax0, ax1 = fig.get_axes()
            ax0.plot(self.plot_freq_axis, values.real, label=label)
            ax1.plot(self.plot_freq_axis, values.imag, label=label)

        plt.draw()

        return selected_meas

    @action("Phase difference", group="Phase plots")
    def plot_meas_phi_diff(self):
        label = self.plot_settings.label
        plot_range = self.plot_settings.plot_range

        sam_meas0 = self.get_selected_measurement(also_return_ref=False)
        sam_meas1 = self.dataset.get_measurement_from_point(*self.comparison_point)

        simple_eval_res0 = self.dataset.windowing_eval(sam_meas0)
        simple_eval_res1 = self.dataset.windowing_eval(sam_meas1)
        phi0 = simple_eval_res0["phi_corrected"]
        phi1 = simple_eval_res1["phi_corrected"]
        phi_diff = phi0-phi1

        f_idx_range = f_axis_idx_map(self.dataset.freq_axis, plot_range)
        plt.figure("Phi difference")
        plt.plot(self.dataset.freq_axis[f_idx_range], phi_diff[f_idx_range], label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase difference (rad)")

    @action("Reference noise", group="Plots")
    def plot_frequency_noise(self):
        if len(self.measurements["refs"]) == 0:
            logging.warning("No measurements classified as reference")
            return
        ref_meas_set = self.measurements["refs"]

        freq_axis = self.dataset.freq_axis

        ampl_arr_db = np.zeros((len(ref_meas_set), len(freq_axis)))
        for i, ref in enumerate(ref_meas_set):
            ref_td, ref_fd = self.dataset.get_data(ref, domain=Domain.Both)
            ampl_arr_db[i] = 20*np.log10(np.abs(ref_fd[:, 1]))

        plt.figure("Amplitude noise (standard deviation of all refs)")
        plt.plot(freq_axis, np.std(ampl_arr_db, axis=0))
        plt.xlabel(f"Frequency (THz)")
        plt.ylabel("Amplitude (dB)")

    @action("System stability", group="Stability plots")
    def plot_system_stability(self):
        stability_data = self.get_stability_data()
        if stability_data is None:
            return

        meas_set = stability_data["meas_set"]
        meas_times = stability_data["meas_times"]
        angle_arr = stability_data["angle_arr"]
        relative_delay = stability_data["relative_delay"]
        ampl_arr = stability_data["ampl_arr"]
        zero_crossing = stability_data["zero_crossing"]
        spec_similarity = stability_data["spec_similarity"]

        t0 = meas_set[0].meas_time
        mt_unit = meas_times.units

        meas_interval = np.mean(np.diff(meas_times)).to("min")
        angle_arr = np.unwrap(angle_arr)

        minima = local_minima_1d(angle_arr, en_plot=False)
        period, period_std = minima[1] * meas_interval, minima[2] * meas_interval

        relative_delay *= 1000

        # correction
        # angle_arr -= 2*np.pi*self.dataset.freq_axis[f_idx]*(zero_crossing/1000)

        selected_freq_ = self.sel_freq_range[0].magnitude
        abs_p_shifts = np.abs(np.diff(relative_delay))
        logging.info(f"Mean pulse shift: {np.round(np.mean(abs_p_shifts), 2)} fs")
        max_diff_0x, min_diff_0x = np.max(abs_p_shifts), np.min(abs_p_shifts)
        logging.info(f"Largest/smallest shift: {np.round(max_diff_0x, 2)}/{np.round(min_diff_0x, 2)} fs")

        max_diff, argmax_diff = np.max(np.diff(angle_arr)), np.argmax(np.diff(angle_arr))
        phase_str = f"Largest phase jump: {np.round(max_diff, 2)} rad"
        phase_str += f" (time: {np.round(meas_times[argmax_diff], 2)})"
        phase_str += f" (at {selected_freq_} THz)"
        logging.info(phase_str)

        avg_amp_change = np.mean(np.abs(np.diff(ampl_arr)))
        max_amp_change = np.max(np.diff(ampl_arr))

        logging.info(f"Largest amplitude change: {np.round(max_amp_change, 2)} (Arb. u.)")
        logging.info(f"Mean absolute amplitude change: {np.round(avg_amp_change, 2)} (Arb. u.)")
        logging.info(f"Mean measurement interval: {np.round(meas_interval, 2)}")
        logging.info(f"Period estimation: {np.round(period, 2)}±{np.round(period_std, 2)}.")

        plt.figure("rfft of the phase change")
        phi_fft = np.fft.rfft(angle_arr)
        phi_fft_f = np.fft.rfftfreq(len(angle_arr), d=meas_interval.to("h").magnitude)

        plt.plot(phi_fft_f[1:], np.abs(phi_fft)[1:])
        plt.xlabel("Frequency (1/hour)")
        plt.ylabel("Magnitude")

        angle_change = angle_arr[0] - angle_arr
        ampl_change = ampl_arr[0] - ampl_arr
        if self.plot_settings.stability_plot_rel_change:
            ampl_change = 100 * ampl_change / ampl_arr[0]
            angle_change = 100 * angle_change / angle_arr[0]

        from random import choice
        idx = choice(range(len(meas_set) - 1))
        meas0, meas1 = meas_set[idx], meas_set[idx + 1]
        meas0_fd = self.dataset.get_data(meas0, domain=Domain.Frequency)
        meas1_fd = self.dataset.get_data(meas1, domain=Domain.Frequency)
        phi0, phi1 = np.angle(meas0_fd[:, 1]), np.angle(meas1_fd[:, 1])
        amp0, amp1 = np.abs(meas0_fd[:, 1]), np.abs(meas1_fd[:, 1])
        w = 2 * np.pi * self.dataset.freq_axis

        plt.figure("Specific pulse shift")
        plt.title(f"Pulse shift between reference number {idx} and {idx+1}")
        plt.plot(self.dataset.freq_axis, 1e3 * (phi0 - phi1) / w, label=idx)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Time (fs)")

        plt.figure("Interpolation zero crossing")
        plt.plot(meas_times, zero_crossing)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel("Time (ps)")

        plt.figure("Amp change")
        plt.plot(self.dataset.freq_axis, amp0 - amp1)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude change (Arb. u.)")

        plt.figure("Reference delay")
        plt.title(f"Reference delay\n(relative to first measurement)")
        plt.plot(meas_times, relative_delay, label="Pulse shift")
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel(r"$\Delta$t (fs)")

        plt.figure("Reference delay change")
        plt.title(f"Reference delay change")
        plt.plot(meas_times[1:], abs_p_shifts, label=t0)
        # phase_change = np.abs(np.diff(angle_change))
        # plt.plot(meas_times[1:], 1e3*phase_change/(2*3.1415*selected_freq_), label=t0)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel(r"$\Delta (\Delta$t) (fs)")

        plt.figure("Stability amplitude")
        plt.title(f"Change of amplitude of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, ampl_change)
        plt.xlabel(f"Measurement time ({mt_unit})")
        if self.plot_settings.stability_plot_rel_change:
            plt.ylabel(r"$\Delta$A (%)")
        else:
            plt.ylabel(r"$\Delta$A (arb. u.)")

        plt.figure("Stability phase")
        plt.title(f"Change of phase of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, angle_change)
        plt.xlabel(f"Measurement time ({mt_unit})")
        if self.plot_settings.stability_plot_rel_change:
            plt.ylabel(r"$\Delta \phi$ (%)")
        else:
            plt.ylabel(r"$\Delta \phi$ (rad)")

        plt.figure("Time between reference measurements")
        plt.title(f"Time between reference measurements")
        plt.plot(meas_times[1:], np.diff(meas_times) * 3600)
        plt.ylabel("Time difference (s)")
        plt.xlabel(f"Measurement time ({mt_unit})")

        plt.figure("Spectral similarity")
        plt.title("Spectral similarity compared to the first measurement")
        plt.plot(meas_times, spec_similarity)
        plt.ylim((-0.05, 1.05))
        plt.ylabel("1 + ln|pears_r|")
        plt.xlabel(f"Measurement time ({mt_unit})")

        ret = {"meas_times": meas_times, "relative_delay": relative_delay}

        climate_plot_ret = self.plot_climate(mt_unit)
        if climate_plot_ret is None:
            return ret
        else:
            climate_meas_times, climate_value_dict = climate_plot_ret

        # climate_value_dict = key: sensor_id, dict[key] = [original_val_arr, smooth_val_arr]

        thz_meas_times = [meas.meas_time for meas in meas_set]
        plotted_climate_vals = {k: np.zeros(len(meas_set)) for k in climate_value_dict}
        for thz_meas_idx, thz_meas_time in enumerate(thz_meas_times):
            best_fit = (None, np.inf)
            for climate_meas_idx, climate_meas_time in enumerate(climate_meas_times):
                meas_time_diff = np.abs((climate_meas_time - thz_meas_time).total_seconds())
                if meas_time_diff < best_fit[1]:
                    best_fit = (climate_meas_idx, meas_time_diff)

            for k in climate_value_dict:
                plotted_climate_vals[k][thz_meas_idx] = climate_value_dict[k][1][best_fit[0]]

        plt.figure("pearsonplot")
        plt.xlabel(f"Time ({mt_unit})")
        for k in plotted_climate_vals:
            shift_arr = np.arange(-100, 100, 1)
            r_vals = np.zeros(len(shift_arr))
            # for idx_shift in np.arange(-70, 71, 1):
            for i, idx_shift in enumerate(shift_arr):
                r = pearsonr(np.diff(plotted_climate_vals[k]), np.roll(relative_delay[1:], idx_shift))
                # r = pearsonr(plotted_climate_vals[k], np.roll(relative_delay, idx_shift))
                r_vals[i] = r.statistic

            argmax = np.argmax(np.abs(r_vals))

            highest_correlation = [r_vals[argmax], shift_arr[argmax]]

            max_corr_val = np.round(highest_correlation[0], 3)
            time_shift = np.round(highest_correlation[1] * meas_interval, 2)
            msg = f"Pearson r (climate quantity, pulse delay) for {k}: {max_corr_val}"
            msg += f" when shifted by {time_shift}"
            logging.info(msg)

            plt.plot(shift_arr * meas_interval.magnitude, r_vals, label=k)

        label_map = self.plot_settings.redp_sensor_labels
        plt.figure("Climate correlation plot")
        for k in plotted_climate_vals:
            label = label_map.get(k, k)
            x = np.gradient(plotted_climate_vals[k], 0.012186554258538694)
            x = plotted_climate_vals[k]
            if "0" in k:
                y = relative_delay
                p = np.polyfit(x, y, 1)
                y = x * p[0] + p[1]
                plt.plot(x, y, label=f"linear fit {label}")
            plt.scatter(x, relative_delay, label=label)
        plt.ylabel("Pulse shift (fs)")
        plt.xlabel("Temperature (°C)")

        return ret

    @action("Climate", group="Stability plots")
    def plot_climate(self, time_unit=None):
        climate_log_file = self.plot_settings.climate_file
        if not climate_log_file.is_file():
            logging.info(f"The path {climate_log_file} is not a file. Check plotting settings for climate plot")
            return None

        log_file = self.plot_settings.climate_file

        full_log_path = self.dataset.find_climate_log_file(log_file)
        if not full_log_path:
            logging.info("No matching climate logfile found")
            return None
        else:
            logging.info(f"Using climate logfile: {full_log_path}")

        quantity = self.settings.plot_opt.climate_quantity
        is_rp_log = False
        if "pitaya" in str(full_log_path):
            is_rp_log = True
            if quantity == ClimateQuantity.Humidity:
                logging.info("The Redpitaya does not record humidity")
        temp_sensor_idx = self.plot_settings.temp_sensor_idx

        def read_log_file(log_file_):
            meas_time_, temp_, humidity_ = [], [], []
            if is_rp_log:
                rp_data = pd.read_csv(log_file_)
                meas_time_ = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in rp_data.iloc[:, 0]]
                if temp_sensor_idx < 0:
                    temp_ = rp_data.iloc[:, 1:]
                else:
                    temp_ = rp_data.iloc[:, temp_sensor_idx+1]
                humidity_ = np.zeros_like(meas_time_)
            else:
                def read_line(line_):
                    parts = line_.split(" ")
                    t = datetime.strptime(f"{parts[0]} {parts[1]}", '%Y-%m-%d %H:%M:%S')
                    return t, float(parts[4]), float(parts[-3])

                with open(log_file_) as file:
                    for i, line in enumerate(file):
                        if "nan" in line:
                            continue
                        if i % 15: # Sampling time: 2 sec (= 0.5 Hz) -> 300 * 2 = 600 sec
                            continue
                        try:
                            res = read_line(line)
                            meas_time_.append(res[0])
                            temp_.append(res[1])
                            humidity_.append(res[2])
                        except IndexError:
                            continue

            return meas_time_, np.array(temp_), np.array(humidity_)

        meas_time, temp, humidity = read_log_file(full_log_path)

        if self.measurements["all"] is not None:
            t0 = self.measurements["all"][0].meas_time
            tf = self.measurements["all"][-1].meas_time
            tf_idx = np.argmin(np.abs([(tf - t).total_seconds() for t in meas_time]))
        else:
            t0 = meas_time[0]
            tf_idx = len(meas_time)

        meas_time_diff = Q_(np.array([(t - t0).total_seconds() for t in meas_time]), "s")
        if time_unit is None:
            meas_time_diff = self.set_time_axis_unit(meas_time_diff)
        else:
            meas_time_diff = meas_time_diff.to(time_unit)
        mt_unit = meas_time_diff.units

        if quantity == ClimateQuantity.Temperature:
            quant = temp
            y_label = r"$\theta$ (°C)"
            dy_label = r"$\partial \theta / \partial t_m$" + f" (°C/{mt_unit})"
        else:
            quant = humidity
            y_label = "Humidity (%)"
            dy_label = fr"$\Delta$Humidity (\\%/{mt_unit})"

        if self.plot_settings.clip_climate_data:
            meas_time = meas_time[:tf_idx]
            meas_time_diff = meas_time_diff[:tf_idx]
            quant = quant[:tf_idx]

        sas = (40, 15) # smoothing_average_settings
        quant_values = {}
        if is_rp_log:
            if quant.ndim != 1:
                for i in range(np.shape(quant)[1]):
                    vals = quant[:, i]
                    smooth_vals = moving_average(vals, iterations=sas[0], n=sas[1])
                    quant_values[f"Redp idx {i}"] = np.array([vals, smooth_vals])
            else:
                vals = quant
                smooth_vals = moving_average(vals, iterations=sas[0], n=sas[1])
                quant_values[f"Redp idx {temp_sensor_idx}"] = np.array([vals, smooth_vals])
        else:
            vals = quant
            smooth_vals = moving_average(vals, iterations=sas[0], n=sas[1])
            quant_values["Arduino"] = np.array([vals, smooth_vals])

        if self.plot_settings.subtract_mean:
            for k in quant_values:
                offset = np.mean(quant_values[k][0])
                std_quant = np.std(quant_values[k][0])
                quant_values[k][0] -= offset
                quant_values[k][1] -= offset
                print(k, offset, std_quant)

        line_labels = self.plot_settings.redp_sensor_labels

        line_colors = ["r", "b", "g", "c", "m", "y", "k"]
        stability_figs = ["Reference zero crossing", "Stability amplitude", "Stability phase"]
        stability_figs.extend(["Reference delay", "Interpolation zero crossing"])
        for fig_label in stability_figs:
            if plt.fignum_exists(fig_label):
                old_fig = plt.figure(fig_label)
                ax_list = old_fig.get_axes()
                lines = ax_list[0].get_lines()
                plt.close(fig_label)

                fig, (ax0, ax1, ax2) = plt.subplots(3, 1, num=fig_label,
                                               sharex=True, gridspec_kw={'hspace': 0})
                ax0.tick_params(bottom=False, labelbottom=False)
                ax1.tick_params(bottom=False, labelbottom=False)

                for i, k in enumerate(quant_values):
                    c = line_colors[i]
                    label = line_labels.get(k, k)
                    # label = None
                    ax0.plot(meas_time_diff, quant_values[k][0], c=c, alpha=0.15)
                    ax0.plot(meas_time_diff, quant_values[k][1], c=c, label=label)
                # ax0.set_yticks([-0.25, 0, 0.25])
                ax0.set_ylabel(y_label)
                ax0.grid(True)

                for i, k in enumerate(quant_values):
                    dqdt = np.gradient(quant_values[k][1], np.mean(np.diff(meas_time_diff)))
                    c = line_colors[i]
                    label = line_labels.get(k, k)
                    label = None
                    ax1.plot(meas_time_diff, dqdt, c=c, label=label)
                # ax1.set_yticks([-2.6, 0, 1.6])
                ax1.set_ylabel(dy_label)
                ax1.grid(True)
                ax1.tick_params(axis="y")

                c = "black"
                for line in lines:
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    # c = line.get_color()
                    ax2.plot(x_data, y_data, color=c, label=line.get_label())
                if "delay" in fig_label:
                    # ax2.set_yticks([0, -50, -100])
                    pass
                ax2.tick_params(axis="y", colors=c)
                ax2.set_ylabel(ax_list[0].get_ylabel(), c=c)
                # ax0.grid(c="blue")
                ax2.grid(True)
                ax2.set_xlabel(f"Measurement time $t_m$ ({mt_unit})")

                if "delay" in fig_label:
                    h0, l0 = ax0.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()

                    ax0.legend(h0 + h2, l0 + l2, loc="upper right", framealpha=0.910)

                fig.align_ylabels([ax0, ax1, ax2])
                axes = [ax0, ax1, ax2]
                labels = ["a)", "b)", "c)"]
                labels = []

                box_style = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.95, lw=0)
                for ax, label in zip(axes, labels):
                    ax.text(0.010, 0.95, label, transform=ax.transAxes,
                            fontsize=34, fontweight="bold", va="top", ha="left", bbox=box_style)

                #ymin_0, ymax_0 = ax0.get_ylim()
                #ax0.set_ylim(bottom=ymin_0, top=ymax_0 - (ymax_0 - ymin_0) * 0.05)
                #ymin_1, ymax_1 = ax1.get_ylim()
                #ax1.set_ylim(bottom=ymin_1 + (ymax_1 - ymin_1) * 0.05, top=ymax_1)

        if not plt.fignum_exists(stability_figs[0]):
            fig, ax1 = plt.subplots(num="Climate plot")
            for i, k in enumerate(quant_values):
                ax1.scatter(meas_time_diff, quant_values[k][0], c=line_colors[i], alpha=0.11, label=f"Start: {t0}" + k)
                ax1.scatter(meas_time_diff, quant_values[k][1], c=line_colors[i])
            ax1.set_xlabel(f"Measurement time ({mt_unit})")
            ax1.set_ylabel(y_label)

        return meas_time, quant_values

    @action("Stability difference", group="Stability plots")
    def system_stability_diff_plot(self):
        kw_filter_str = "" # original: "-sub-": monitoring_pulse_mod\set1
        system_stab_res_refs = self.get_stability_data()
        x = system_stab_res_refs["meas_times"]
        y_ref = system_stab_res_refs["relative_delay"]

        self.settings.pp_opt.win_start = 11
        system_stab_res_mon_pulse0 = self.get_stability_data(meas_set_kw=kw_filter_str)
        if system_stab_res_mon_pulse0 is None:
            return

        self.settings.pp_opt.win_start = 27
        system_stab_res_mon_pulse1 = self.get_stability_data(meas_set_kw=kw_filter_str)
        if system_stab_res_mon_pulse1 is None:
            return

        xp = system_stab_res_mon_pulse0["meas_times"]
        fp = system_stab_res_mon_pulse0["relative_delay"]
        y_pulse0 = np.interp(x, xp, fp)

        xp = system_stab_res_mon_pulse1["meas_times"]
        fp = system_stab_res_mon_pulse1["relative_delay"]
        y_pulse1 = np.interp(x, xp, fp)

        delay_difference_pulse0 = y_pulse0 - y_ref
        # offset_pulse0 = np.mean(delay_difference_pulse0[100:])
        offset_pulse0 = 0
        delay_difference_pulse1 = y_pulse1 - y_ref
        # offset_pulse1 = np.mean(delay_difference_pulse1[100:])
        offset_pulse1 = 0
        # print(offset_pulse0, offset_pulse1)

        y_pulse0 = y_pulse0 - offset_pulse0
        y_pulse1 = y_pulse1 - offset_pulse1

        y_mean = (y_pulse0 + y_pulse1) / 2

        residual_pulse0 = np.sum((y_pulse0 - y_ref)**2) / len(y_ref)
        residual_pulse1 = np.sum((y_pulse1 - y_ref) ** 2) / len(y_ref)
        residual_mean = np.sum((y_mean - y_ref) ** 2) / len(y_ref)
        # print(residual_pulse0, residual_pulse1, residual_mean)

        plt.figure("Delay interpolation")
        # plt.plot(x, y_pulse0, label="Delay monitor pulse0 (interp)")
        # plt.plot(x, y_pulse1, label="Delay monitor pulse1 (interp)")
        plt.plot(x, y_mean, label="Delay monitor mean pulse 0 and 1")
        plt.plot(x, y_ref, label="Delay reference")
        plt.xlabel(f"Measurement time (unit?)")
        plt.ylabel("Time (fs)")

        plt.figure("Delay difference")
        plt.plot(x, delay_difference_pulse0, label="difference y_mon_pulse0 - y_ref")
        plt.plot(x, delay_difference_pulse1, label="difference y_mon_pulse1 - y_ref")
        plt.xlabel(f"Measurement time (unit?)")
        plt.ylabel("Time (fs)")

    def on_image_click(self, event):
        if not self.enable_img_interaction:
            return
        if event.inaxes is None:
            return
        toolbar = event.canvas.toolbar
        if toolbar is not None and toolbar.mode != "":
            return

        if event.button is MouseButton.LEFT:
            dx = self.dataset.shape_properties["dx"]
            dy = self.dataset.shape_properties["dy"]
            x = round_dx(event.xdata, dx)
            y = round_dx(event.ydata, dy)
            point = (x, y)
            if self.selected_quantity == QuantityEnum.P2P:
                plotted_meas = self.plot_waveform(point)
            else:
                plotted_meas = self.plot_selected_quantity(point)


        self._plot_meas_on_image(plotted_meas)

        self.plt_show()


    @action("Plot image", group=image_grp, priority=1)
    def plot_image(self):
        shape_properties = self.img_shape

        if not self.confine_to_extent:
            img_extent = shape_properties["extent"]
            w0, w1, h0, h1 = [0, shape_properties["w"], 0, shape_properties["h"]]
        else:
            img_extent = [*(self.img_extent_x_range.magnitude), *(self.img_extent_y_range.magnitude)]
            dx, dy = shape_properties["dx"], shape_properties["dy"]
            w0, w1 = (int((img_extent[0] - shape_properties["extent"][0]) / dx),
                      int((img_extent[1] - shape_properties["extent"][0]) / dx))
            h0, h1 = (int((img_extent[2] - shape_properties["extent"][2]) / dy),
                      int((img_extent[3] - shape_properties["extent"][2]) / dy))

        self.grid_vals = self._calc_grid_vals()

        shown_grid_vals = self.grid_vals.real
        shown_grid_vals = shown_grid_vals[w0:w1, h0:h1]
        shown_grid_vals = self._exclude_pixels(shown_grid_vals)

        if self.plot_settings.log_scale:
            shown_grid_vals = np.log10(shown_grid_vals)

        fig = plt.figure(self.image_fig_num)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2)

        if self.plot_settings.en_cbar_lim:
            cbar_min, cbar_max = self.plot_settings.cbar_lim
        else:
            cbar_min = np.min(shown_grid_vals)
            cbar_max = np.max(shown_grid_vals)

        if self.plot_settings.log_scale:
            cbar_min = np.log10(cbar_min)
            cbar_max = np.log10(cbar_max)

        axes_extent = (float(img_extent[0] - shape_properties["dx"] / 2),
                       float(img_extent[1] + shape_properties["dx"] / 2),
                       float(img_extent[2] - shape_properties["dy"] / 2),
                       float(img_extent[3] + shape_properties["dy"] / 2))
        img_ = ax.imshow(shown_grid_vals.transpose((1, 0)),
                         vmin=cbar_min, vmax=cbar_max,
                         origin="lower",
                         cmap=plt.get_cmap(self.plot_settings.color_map.value),
                         extent=axes_extent,
                         interpolation=self.plot_settings.pixel_interpolation.value
                         )

        if self.plot_settings.invert_x:
            ax.invert_xaxis()
        if self.plot_settings.invert_y:
            ax.invert_yaxis()

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        quantity_label = self.quantity_label

        img_title_option = str(self.plot_settings.img_title)
        ax.set_title(" ".join([quantity_label, img_title_option]))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(img_, cax=cax)
        cbar.set_ticks(np.round(np.linspace(cbar_min, cbar_max, 4), 3))

        if self.plot_settings.en_cbar_label:
            quant = self.selected_quantity.value
            cbar_label = quant.label +" "+  quant.unit
            cbar.set_label(cbar_label, rotation=270, labelpad=30)

        plt.connect('button_press_event', self.on_image_click)

        self.img_ax = ax

    def _plot_meas_on_image(self, measurements):
        if not plt.fignum_exists(self.image_fig_num):
            return
        if measurements is None:
            return

        if isinstance(measurements, Measurement):
            measurements = [measurements]

        plt.figure(num=self.image_fig_num)
        img_ax = self.img_ax

        ext = self.img_shape["extent"]

        any_in_extent = False
        meas_x_coords, meas_y_coords = [], []
        for m in measurements:
            x, y = m.position
            meas_x_coords.append(x)
            meas_y_coords.append(y)

            if (ext[0] < x < ext[1]) * (ext[2] < y < ext[3]):
                any_in_extent = True

        if not any_in_extent:
            logging.info("None of the measurements within the image extent")

        plt_fun = img_ax.scatter
        pnt = plt_fun(meas_x_coords, meas_y_coords, color="black", linewidth=0.4)
        self.drawn_elements["points"].append(pnt)

        plt.draw()

    @action("Plot references on image", group=image_grp)
    def plot_refs_on_image(self):
        self._plot_meas_on_image(self.measurements["refs"])

    @action("Line plot", group=line_plot_grp)
    def plot_line(self):
        start = self.line_start.magnitude
        end = self.line_end.magnitude
        meas_on_line, meas_points = self.dataset.get_arb_line(start, end)

        logging.info("Calculating line values")
        vals = []
        for i, measurement in enumerate(meas_on_line):
            i += 1
            msg = f"{round(100 * i / len(meas_on_line), 2)} % done. "
            msg += f"(Measurement: {i}/{len(meas_on_line)}, {measurement.position} mm)"
            if i == len(meas_on_line):
                msg += "\n"
            logging.info(msg)

            vals.append(self.scalar_func(measurement))

        x_coords, y_coords = [p[0] for p in meas_points], [p[1] for p in meas_points]
        if len(set(x_coords)) == len(meas_points):
            primary_direction = Direction.Horizontal
        else:
            primary_direction = Direction.Vertical

        if primary_direction == Direction.Horizontal:
            fig_num = "x-slice"
            x_label = "x (mm)"
            x_axis_vals = x_coords
        else:
            fig_num = "y-slice"
            x_label = "y (mm)"
            x_axis_vals = y_coords

        fig_num += "_" + self.quantity_label.replace(" ", "_")
        plt.figure(fig_num)
        plt.title(f"Line scan ({primary_direction.name})")
        plt.xlabel(x_label)
        plt.ylabel(self.quantity_label)

        unit = self.line_start[0].units
        line_label = f"({start[0]}, {start[1]}) to ({end[0]}, {end[1]}) {unit}"

        plt.plot(x_axis_vals, vals, label=line_label)
        plt.legend()
        plt.draw()

        self._plot_meas_on_image(meas_on_line)

        return x_axis_vals, vals, fig_num

    @action("Knife edge", group=line_plot_grp)
    def knife_edge(self):
        if self.selected_quantity != QuantityEnum.PowerInt:
            logging.info("Integrated power must be selected")
            return

        coords, vals, fig_num = self.plot_line()

        pos_axis = np.array(coords)
        sort_order = np.argsort(pos_axis)

        pos_axis = pos_axis[sort_order]
        vals = np.array(vals)[sort_order]

        pos_axis_ordered = pos_axis if np.argmin(vals) > np.argmax(vals) else np.flip(pos_axis)

        def _model(x, p_max, p_offset, w, h0):
            return p_offset + 0.5 * p_max * erfc(np.sqrt(2) * (x - h0) / w)

        def _cost(p):
            return np.sum((vals - _model(pos_axis_ordered, *p)) ** 2)

        slope_pos = pos_axis_ordered[np.argmax(np.abs(np.diff(vals))) + 1]

        p0 = np.array([np.max(vals), np.min(vals), 0.5, slope_pos])
        bounds = ([p0[0] - 1, p0[0] + 1],
                  [p0[1], p0[1] + 0.01],
                  [p0[2] - 0.4, p0[2] + 2.0],
                  [p0[3] - 2, p0[3] + 2])
        opt_res = shgo(_cost, bounds=bounds)

        popt, pcov = curve_fit(
            _model,
            pos_axis_ordered,
            vals,
            p0=opt_res.x,
            bounds=([bounds[i][0] for i in range(4)], [bounds[i][1] for i in range(4)])
        )

        perr = np.sqrt(np.diag(pcov))

        plt.figure(fig_num)
        plt.scatter(pos_axis, vals, label="Measurement", s=30, c="red", zorder=3)
        plt.plot(pos_axis, _model(pos_axis_ordered, *popt), label="Optimization result")
        plt.plot(pos_axis, _model(pos_axis_ordered, *p0), label="Initial guess", linestyle="--")

        labels = ["$P_{max}$: ", "$P_{offset}$: ", "Beam radius: ", "$h_0$: "]
        units = ["", "", " mm", " mm"]
        s = "\n".join([
            f"{lbl}{val:.2f} ± {err:.2f}{unit}"
            for lbl, val, err, unit in zip(labels, np.abs(popt), perr, units)
        ])

        bbox_props = dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85,
            linewidth=1
        )

        plt.gca().text(
            0.05, 0.95, s,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=18,
            fontfamily="sans-serif",
            bbox=bbox_props,
            zorder=5
        )

        plt.legend()
        plt.draw()

        return popt, pcov

    def save_fig(self, fig_num_, **kwargs):
        save_dir = self.settings.save_settings.path
        filetype = self.settings.save_settings.filetype
        kwargs.setdefault("dpi", self.settings.save_settings.dpi)
        kwargs.setdefault("bbox_inches", self.settings.save_settings.bbox_inches)
        kwargs.setdefault("pad_inches", self.settings.save_settings.pad_inches)

        filename = str(fig_num_)

        suffix = self.settings.save_settings.suffix
        filename = str(fig_num_) if not suffix else str(fig_num_) + "_" + suffix

        fig = plt.figure(fig_num_)
        if filename is None:
            try:
                filename_s = str(fig.canvas.get_window_title())
            except AttributeError:
                filename_s = str(fig.canvas.manager.get_window_title())
        else:
            filename_s = str(filename)

        illegal_chars = ["(", ")"]
        for char in illegal_chars:
            filename_s = filename_s.replace(char, '')
        filename_s.replace(" ", "_")
        full_path = save_dir / (filename_s + str(filetype.value))

        w = self.settings.save_settings.set_size_inches
        fig.set_size_inches(w=w, forward=False)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(full_path, **kwargs)

        return full_path

    @action("Close all figures", group="Show / close plots")
    def close_figures(self):
        for fig_num in plt.get_fignums():
            plt.close(fig_num)

    @action("Save open figures", group="Show / close plots")
    def save_open_figures(self):
        for i, fig_num in enumerate(plt.get_fignums()):
            path = self.save_fig(fig_num)
            logging.info(f"Saved figure {fig_num} to {path}")

    @action("Show plots", group="Show / close plots")
    def plt_show(self, save_file_suffix=None):
        for i, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)
            axes = fig.get_axes()
            for ax in axes:
                leg = ax.get_legend()
                h, labels = ax.get_legend_handles_labels()
                if labels:
                    if leg is not None:
                        ax.legend(h, labels,
                            loc=leg._loc,
                            framealpha=leg.get_frame().get_alpha(),
                        )
                    else:
                        ax.legend(h, labels)

            if self.settings.save_settings.save_plots:
                self.save_fig(fig_num)

            if self.settings.save_settings.only_save_plots:
                if i == 0:
                    logging.info("Showing plots disabled in settings. Only saving if enabled")
                plt.close(fig_num)
                continue

        plt.show()




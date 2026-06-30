from common.components import ComponentBase, action
from common.dataset import DataSet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from common.default_appsettings import Domain, Dist, Direction, ClimateQuantity, QuantityFunc, QuantityEnum
from functools import partial
from common.functions import moving_average, f_axis_idx_map, local_minima_1d
import logging
from datetime import datetime
from pathlib import Path
from scipy.special import erfc
from common.eval_component.shgo import shgo
from traitlets import Float, observe, Integer, Unicode, Enum as TEnum
from common.traits import Q_, Quantity, ValueRange
from mpl_settings import mpl_style_params
import matplotlib as mpl
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

class DataSetPlotter(ComponentBase):

    sel_freq_range = ValueRange(default_value=[Q_(1.000, "THz"), Q_(1.200, "THz")]).tag(
        name="Selected frequency range"
    )
    sel_point = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(name="Selected point (x, y)")
    sel_timestamp = Unicode("").tag(name="Selected timestamp")
    selected_quantity = TEnum(QuantityEnum, default_value=QuantityEnum.P2P).tag(name="Selected quantity")

    def __init__(self, dataset : DataSet, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.img_properties = {}

        self.selected_freq_idx = None

        self.grid_func = None
        self.grid_vals = None
        self.img_ax = None

        self._apply_settings()

    @observe("sel_freq_range")
    def select_freq(self, change):
        self.set_trait("sel_freq_range", change["new"])
        self.selected_freq_idx = self.freq_idx()
        self._update_fig_num()

    @property
    def settings(self):
        return self.dataset.settings

    @property
    def plot_settings(self):
        return self.dataset.settings.plot_opt

    @property
    def measurements(self):
        return self.dataset.measurements

    def _plot_action(self, *args, **kwargs):
        action(*args, **kwargs)
        self.plt_show()

    def _calc_grid_vals(self):
        if self.grid_vals is not None:
            return self.grid_vals

        w, h = self.dataset.properties["shape"]["w"], self.dataset.properties["shape"]["h"]
        grid_vals = np.zeros((w, h), dtype=complex)
        sam_meas = self.measurements["sams"]

        iter_ = tqdm(enumerate(sam_meas), total=len(sam_meas),
                     desc="Evaluating measurements", colour="green")
        for i, measurement in iter_:
            x_idx, y_idx = self._coords_to_idx(*measurement.position)

            grid_vals[x_idx, y_idx] = self.grid_func(measurement)

        return grid_vals

    def _apply_settings(self):
        mpl.rcParams.update(mpl_style_params())

    def freq_idx(self, freq=None):
        if freq is None:
            freq = self.sel_freq_range[0]
        selected_freq_idx = f_axis_idx_map(self.dataset.freq_axis, freq)

        return selected_freq_idx

    @observe("selected_quantity")
    def select_quantity(self, change, label=""):
        quantity = change["new"]

        func_map = self.dataset.func_map

        func_map[QuantityEnum.Power] = partial(self.dataset.power, freq_range=self.selected_freq_idx)
        func_map[QuantityEnum.PeakCnt] = partial(self.dataset.simple_peak_cnt, threshold=2.5)

        if isinstance(quantity, QuantityFunc):
            if not callable(quantity.func):
                logging.warning("Func of Quantity must be callable")
            func = quantity.func
            selected_quantity = quantity
        elif callable(quantity):
            func = quantity
            selected_quantity = QuantityFunc(label=label, func=quantity)
        elif quantity in func_map:
            func = func_map[quantity]
            selected_quantity = quantity.value
        else:
            logging.warning(f"Unknown quantity type: {quantity}")

        def grid_func(meas, func=func):
            sel_freq_idx = self.freq_idx()

            res = np.real(func(meas))
            ndim = np.ndim(res)
            if ndim == 0:
                return res
            elif ndim == 1:
                return res[sel_freq_idx]
            else:
                return res[sel_freq_idx, 1]

        self.grid_func = grid_func
        self.selected_quantity = selected_quantity

        self._update_fig_num()

    def _coords_to_idx(self, x_, y_):
        shape_properties = self.dataset.properties["shape"]
        x, y = shape_properties["x_coords"], shape_properties["y_coords"]
        x_idx, y_idx = np.argmin(np.abs(x_ - x)), np.argmin(np.abs(y_ - y))

        return x_idx, y_idx

    def _idx_to_coords(self, x_idx, y_idx):
        shape_properties = self.dataset.properties["shape"]
        dx, dy = shape_properties["dx"], shape_properties["dy"]

        y = shape_properties["y_coords"][0] + y_idx * dy
        x = shape_properties["x_coords"][0] + x_idx * dx

        return x, y

    def _update_fig_num(self):
        en_freq_label = Domain.Frequency == self.selected_quantity.domain
        fig_num = ""
        if self.plot_settings.fig_label:
            fig_num += self.plot_settings.fig_label + " "
        fig_num += str(self.selected_quantity)

        f1, f2 = int(self.sel_freq_range[0].magnitude * 1e3), int(self.sel_freq_range[1].magnitude * 1e3)
        if np.isclose(self.sel_freq_range[0].magnitude, self.sel_freq_range[1].magnitude):
            fig_num += en_freq_label * f" {f1} GHz"
        else:
            fig_num += en_freq_label * f" {f1}-{f2} GHz"
        fig_num = fig_num.replace(" ", "_")

        self.img_properties["fig_num"] = fig_num + self.dataset.is_sub_dataset * "_subset"

        self._update_quantity_label()

    def _update_quantity_label(self):
        en_freq_label = Domain.Frequency == self.selected_quantity.domain
        if np.isclose(self.sel_freq_range[0].magnitude, self.sel_freq_range[1].magnitude):
            freq_label = f"({self.sel_freq_range[0]})"
        else:
            freq_label = f"({self.sel_freq_range[0]}-{self.sel_freq_range[1]})"

        self.img_properties["quantity_label"] = " ".join([str(self.selected_quantity),
                                                          freq_label * en_freq_label])

    def _is_excluded(self, idx_tuple):
        excl_areas = self.plot_settings.excluded_areas
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
        filtered_grid = grid_vals.copy()
        dims = filtered_grid.shape
        for x_idx in range(dims[0]):
            for y_idx in range(dims[1]):
                if self._is_excluded((x_idx, y_idx)):
                    filtered_grid[x_idx, y_idx] = 0

        return filtered_grid

    def average_area(self, pnt_bot_left, pnt_top_right, label="A1"):
        assert (pnt_bot_left[0] <= pnt_top_right[0]) and (pnt_bot_left[1] <= pnt_top_right[1])

        x_coords, y_coords = self.img_properties["x_coords"], self.img_properties["y_coords"]
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

        meas_cnt = grid_vals.shape[0] * grid_vals.shape[1]
        logging.info(f"Average value of area {label} ({meas_cnt} measurements): {mean_s}±{std_s}")
        logging.info(f"Min: {min_s}, max: {max_s}\n")

        ret = {"mean": mean_val, "std": std_val, "min": min_s, "max": max_s}
        if not plt.fignum_exists(num=self.img_properties["fig_num"]):
            return ret

        plt.figure(self.img_properties["fig_num"])
        ax = self.img_ax

        dx, dy = self.img_properties["dx"], self.img_properties["dy"]
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
        ax.add_patch(rect)

        # decide where to put rect label
        img_extent = self.img_properties["extent"]
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
        ax.text(t_x, t_y, label,
                color="black", fontsize=18, ha="center", va="center", fontweight="bold")

        return ret

    @action("Reference difference", group="Plots")
    def ref_difference_plot(self):
        ref1, ref2 = self.measurements["refs"][11], self.measurements["refs"][16]

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

        sel_freq_idx = self.freq_idx()
        mark_x = [dt1, dt2]
        mark_y = [phi1[sel_freq_idx], phi2[sel_freq_idx]]

        plt.figure("Stability phase")
        plt.scatter(mark_x, mark_y, color="red", s=30, zorder=99)

        # self.plt_show()

    @action("Reference measurement", group="Plots")
    def plot_ref(self, ref_meas_=None, timestamp=None, ref_idx=None):

        fig_num_ext = self.plot_settings.fig_num_ext
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
        plt.figure("Spectrum" + fig_num_ext)
        plt.plot(freq_axis[f_idx_range], y_db, label=label + " (Reference)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")

        plt.figure("Time domain" + fig_num_ext)
        plt.plot(ref_td[:, 0], ref_td[:, 1], label=label + " (Reference)")
        if self.plot_settings.plot_zero_crossing:
            plt.scatter(zero_crossing, 0, label="Zero-crossing", color="red")
        # plt.plot(ref_td[1:, 0], np.diff(np.abs(ref_td[:, 1])), label=label)
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (Arb. u.)")

    @action("Measurement", group="Plots")
    def plot_meas(self, timestamp=None, en_td_plot=True):
        if timestamp is None:
            timestamp = self.sel_timestamp

        plot_opt = self.plot_settings
        label = plot_opt.label
        sub_noise_floor = plot_opt.sub_noise_floor
        td_scale = plot_opt.td_scale
        remove_t_offset = plot_opt.remove_t_offset
        std_limits = plot_opt.err_bar_limits # limits of spatial coordinates to average over, for the err_bars
        en_csv_export = self.settings.save_settings.en_csv_export
        fig_num_ext = plot_opt.fig_num_ext
        plot_range = plot_opt.plot_range
        ref_err_bars = plot_opt.ref_err_bars

        f_idx_range = f_axis_idx_map(self.dataset.freq_axis, plot_range)
        if timestamp:
            logging.info(f"Plotting measurement with timestamp: {timestamp}")
            selected_meas = self.dataset.get_measurement_from_timestamp(timestamp)
            point = selected_meas.position
        else:
            point = self.sel_point
            selected_meas = self.dataset.get_measurement(*point)

        ref_meas = self.dataset.get_nearest_ref(selected_meas)

        q_eval_res = None
        meas_quants = self.dataset.calc_meas_quantities(ref_meas, selected_meas)

        if self.settings.enable_q_eval:
            ana_eval_res = self.dataset.single_layer_eval(selected_meas)

            #q_eval = QSpaceEval(self.settings, meas_quants, ana_eval_res)
            #q_eval_res = q_eval.q_space_eval()

        logging.info(f"Plotting point {point}")
        logging.info(f"Reference measurement: {ref_meas}")
        logging.info(f"Sample measurement: {selected_meas}\n")

        # TODO redo window plotting
        show_win_plot = deepcopy(self.settings.pp_opt.en_plot)
        if self.plot_settings.window:
            self.settings.pp_opt.en_plot = True

        self.settings.pp_opt.fig_label = "ref"
        ref_td, ref_fd = self.dataset.get_data(ref_meas, domain=Domain.Both)

        #ref_fd[:, 1] = np.abs(ref_fd[:, 1]) * np.exp(-1j*np.angle(ref_fd[:, 1]))
        #ref_td = do_ifft(ref_fd, conj=False)
        self.settings.pp_opt.fig_label = "sam"

        sam_td, sam_fd = meas_quants["sam_td"], meas_quants["sam_fd"]
        ref_td, ref_fd = meas_quants["ref_td"], meas_quants["ref_fd"]

        if self.plot_settings.shift_sam2ref:
            shift_t = np.abs(np.argmax(ref_td[:, 1]) - np.argmax(sam_td[:, 1]))
            sam_td[:, 1] = np.roll(sam_td[:, 1], -shift_t)

        self.settings.pp_opt.en_plot = show_win_plot

        if remove_t_offset:
            sam_td[:, 0] -= sam_td[0, 0]

        # TODO is this needed? Get error bars from arr_stat function ?
        err_bar_range = None
        if std_limits:
            meas_line, coords = self.dataset.get_line(y=0, limits=std_limits)
            if meas_line:
                absorbance_arrs = []
                for meas in meas_line:
                    sam_fd_line = self.dataset.get_data(meas, domain=Domain.Frequency)
                    t = sam_fd_line[:, 1] / ref_fd[:, 1]
                    absorbance_arrs.append(20*np.log10(np.abs(1/t)))
                err_bar_range = np.std(absorbance_arrs, axis=0)
            else:
                err_bar_range = np.zeros(len(self.dataset.freq_axis))
        if ref_err_bars:
            all_ref_meas = self.measurements["refs"]
            ref_meas_first, ref_meas_last = all_ref_meas[0], all_ref_meas[-1]
            ref_fd_first = self.dataset.get_data(ref_meas_first, domain=Domain.Frequency)
            ref_fd_last = self.dataset.get_data(ref_meas_last, domain=Domain.Frequency)

            if std_limits:
                meas_line, coords = self.dataset.get_line(y=0, limits=std_limits)
            else:
                meas_line = [selected_meas]
            if meas_line:
                absorbance_arrs = []
                for meas in meas_line:
                    sam_fd_line = self.dataset.get_data(meas, domain=Domain.Frequency)
                    t_first = sam_fd_line[:, 1] / ref_fd_first[:, 1]
                    t_last = sam_fd_line[:, 1] / ref_fd_last[:, 1]
                    absorbance_arrs.append(20 * np.log10(np.abs(1 / t_first)))
                    absorbance_arrs.append(20 * np.log10(np.abs(1 / t_last)))

                err_bar_range = np.std(absorbance_arrs, axis=0)
                err_bar_range = np.max(absorbance_arrs, axis=0) - np.min(absorbance_arrs, axis=0)
            else:
                err_bar_range = np.zeros(len(self.dataset.freq_axis))

        freq_axis = ref_fd[:, 0].real

        t = sam_fd[:, 1] / ref_fd[:, 1]
        absorb = np.abs(1/t)

        simple_eval_res = self.dataset.single_layer_eval(selected_meas)
        phi = simple_eval_res["phi"]
        phi_corrected = simple_eval_res["phi_corrected"]
        refr_idx = simple_eval_res["refr_idx"]

        # TODO FIX correct return of measurement quantities and calculated quantities
        if q_eval_res is None:
            ret = {"freq_axis": freq_axis, "absorb": absorb, "t": t, "ref_fd": ref_fd, "sam_fd": sam_fd, "phi": phi,
                   "phi_corrected": phi_corrected, "t_amplitude": np.abs(t)}
        else:
            ret = q_eval_res
            self.dataset.print_ret(ret, label)
        if en_csv_export:
            self.dataset.export_as_csv(ret, file_app=label)

        phi = phi[f_idx_range]
        phi_corrected = phi_corrected[f_idx_range]
        refr_idx = refr_idx[f_idx_range]
        alph = self.dataset.absorption_coef(selected_meas)

        if not self.dataset.plotted_ref:
            self.plot_ref(ref_meas)
            self.dataset.plotted_ref = True

        if not label:
            label = f"(x,y)=({point[0]}, {point[1]})"

        freq_axis = sam_fd[:, 0].real
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum" + fig_num_ext)
        y_db = (20 * np.log10(np.abs(sam_fd[f_idx_range, 1])) - noise_floor).real
        plt.plot(freq_axis[f_idx_range], y_db, label=label)

        plt.figure("Phase correction comparison" + fig_num_ext)
        plt.plot(freq_axis[f_idx_range], phi, label=label + " (Original)", ls="dashed")
        plt.plot(freq_axis[f_idx_range], phi_corrected, label=label + " (Corrected)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")

        plt.figure("Phase" + fig_num_ext)
        plt.plot(freq_axis[f_idx_range], phi_corrected, label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")

        plt.figure("Phase slope" + fig_num_ext)
        plt.plot(freq_axis[f_idx_range][:-1], np.diff(phi_corrected), label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad/THz)")

        plt.figure("Time domain" + fig_num_ext)
        td_label = label
        if not np.isclose(td_scale, 1):
            td_label += f"\n(Amplitude x {td_scale})"
        if en_td_plot:
            plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=td_label)

        if not plt.fignum_exists("Amplitude transmission" + fig_num_ext):
            plt.figure("Amplitude transmission")
            plt.xlabel("Frequency (THz)")
            plt.ylabel(r"Amplitude transmission")
            plt.ylim((-0.05, 1.10))
        else:
            plt.figure("Amplitude transmission" + fig_num_ext)

        plt.plot(freq_axis[f_idx_range], (1/absorb[f_idx_range]), label=label)

        plt.figure("Absorbance" + fig_num_ext)
        y = 20*np.log10(absorb[f_idx_range])
        plt.plot(freq_axis[f_idx_range], y, label=label)
        if std_limits:
            lower = y - err_bar_range[f_idx_range]
            upper = y + err_bar_range[f_idx_range]
            plt.fill_between(freq_axis[f_idx_range], lower, upper, alpha=0.3)

        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorbance (dB)")

        ri_fignum = "Refractive index" + fig_num_ext
        if not plt.fignum_exists(ri_fignum):
            fig, (ax0, ax1) = plt.subplots(2, 1, num=ri_fignum, sharex=True, gridspec_kw={'hspace': 0})
            ax1.set_xlabel("Frequency (THz)")
            ax0.set_ylabel("Refractive index (Real)")
            ax1.set_ylabel("Refractive index (Imag)")
        else:
            fig = plt.figure(ri_fignum)
            ax0, ax1 = fig.get_axes()
        ax0.plot(freq_axis[f_idx_range], refr_idx.real, label=label)
        ax1.plot(freq_axis[f_idx_range], refr_idx.imag, label=label)

        plt.figure("Absorption coefficient" + fig_num_ext)
        plt.plot(freq_axis[f_idx_range], alph[f_idx_range], label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorption coefficient (1/cm)")

        if self.dataset.sub_dataset is not None:
            sigma = self.dataset.conductivity(selected_meas)
            plt.figure("Conductivity" + fig_num_ext)
            plt.title(label)
            plt.plot(freq_axis[f_idx_range], sigma[f_idx_range].real, label="Real part")
            plt.plot(freq_axis[f_idx_range], sigma[f_idx_range].imag, label="Imaginary part")
            # plt.ylim((-1e3, 1.5e5))
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Conductivity (S/cm)")

        return ret, meas_quants

    def plot_meas_phi_diff(self, pnt0, pnt1, label=""):
        plot_range = self.plot_settings.plot_range

        sam_meas0 = self.dataset.get_measurement(*pnt0)
        ref_meas0 = self.dataset.get_nearest_ref(sam_meas0)

        sam_meas1 = self.dataset.get_measurement(*pnt1)

        ref_fd = self.dataset.get_data(ref_meas0, domain=Domain.Frequency)
        freq_axis = ref_fd[:, 0].real

        simple_eval_res0 = self.dataset.single_layer_eval(sam_meas0)
        simple_eval_res1 = self.dataset.single_layer_eval(sam_meas1)
        phi0 = simple_eval_res0["phi_corrected"]
        phi1 = simple_eval_res1["phi_corrected"]
        phi_diff = phi0-phi1

        f_idx_range = f_axis_idx_map(self.dataset.freq_axis, plot_range)
        plt.figure("Phi difference")
        plt.plot(freq_axis[f_idx_range], phi_diff[f_idx_range], label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase difference (rad)")

    @action("Reference noise", group="Plots")
    def plot_frequency_noise(self):
        ref_meas_set = self.measurements["refs"]
        freq_axis = self.dataset.freq_axis

        ampl_arr_db = np.zeros((len(ref_meas_set), len(freq_axis)))
        for i, ref in enumerate(ref_meas_set):
            ref_td, ref_fd = self.dataset.get_data(ref, domain=Domain.Both)
            ampl_arr_db[i] = 20*np.log10(np.abs(ref_fd[:, 1]))


        plt.figure("Amplitude noise")
        #plt.title(f"Amplitude of reference measurement at {selected_freq_} THz")
        plt.plot(freq_axis, np.std(ampl_arr_db, axis=0))

        plt.xlabel(f"Frequency (THz)")
        plt.ylabel("Amplitude (dB)")

    @action("System stability", group="Plots")
    def plot_system_stability(self, climate_log_file=None, meas_set_kw=None):
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

        selected_freq_ = self.sel_freq_range[0].magnitude
        f_idx = np.argmin(np.abs(self.dataset.freq_axis - selected_freq_))

        ampl_arr, angle_arr, relative_delay, spec_similarity = np.zeros((4, len(meas_set)))
        t0 = meas_set[0].meas_time

        meas_times = np.array([self.dataset.meas_time_diff(meas_set[0], m) for m in meas_set])

        if meas_times.max() < 5 / 60:
            conv_factor = 3600
            mt_unit = "seconds"
        elif 5 / 60 <= meas_times.max() < 0.5:
            conv_factor = 60
            mt_unit = "minutes"
        else:
            conv_factor = 1
            mt_unit = "h"

        meas_times *= conv_factor

        zero_crossing = np.zeros(len(meas_set))
        for i, meas_ in enumerate(meas_set):
            ref_td, ref_fd = self.dataset.get_data(meas_, domain=Domain.Both)

            zero_crossing[i] = self.dataset.get_zero_crossing(meas_)
            relative_delay[i] = self.dataset.delay_from_phaseslope(meas_set[0], meas_)
            ampl_arr[i] = np.sum(np.abs(ref_fd[f_idx, 1]))
            angle_arr[i] = -np.angle(ref_fd[f_idx, 1])
            spec_similarity[i] = self.dataset.spectral_similarity(meas_set[0], meas_)

        meas_interval = np.mean(np.diff(meas_times))
        angle_arr = np.unwrap(angle_arr)

        minima = local_minima_1d(angle_arr, en_plot=False)
        period, period_std = minima[1] * meas_interval * 60, minima[2] * meas_interval * 60

        # relative_delay = (relative_delay - relative_delay[0]) * 1000
        relative_delay *= 1000

        # correction
        # angle_arr -= 2*np.pi*self.dataset.freq_axis[f_idx]*(zero_crossing/1000)

        abs_p_shifts = np.abs(np.diff(relative_delay))
        logging.info(f"Mean pulse shift: {np.round(np.mean(abs_p_shifts), 2)} fs")
        max_diff_0x, min_diff_0x = np.max(abs_p_shifts), np.min(abs_p_shifts)
        logging.info(f"Largest/smallest shift: {np.round(max_diff_0x, 2)}/{np.round(min_diff_0x, 2)} fs")

        max_diff, argmax_diff = np.max(np.diff(angle_arr)), np.argmax(np.diff(angle_arr))
        phase_str = f"Largest phase jump: {np.round(max_diff, 2)} rad"
        phase_str += f" (time: {np.round(meas_times[argmax_diff], 2)} {mt_unit})"
        phase_str += f" (at {selected_freq_} THz)"
        logging.info(phase_str)

        avg_amp_change = np.mean(np.abs(np.diff(ampl_arr)))
        max_amp_change = np.max(np.diff(ampl_arr))

        logging.info(f"Largest amplitude change: {np.round(max_amp_change, 2)} (Arb. u.)")
        logging.info(f"Mean absolute amplitude change: {np.round(avg_amp_change, 2)} (Arb. u.)")
        logging.info(f"Mean measurement interval: {np.round(meas_interval * 3600, 2)} sec.")
        logging.info(f"Period (estimation): {np.round(period, 2)}±{np.round(period_std, 2)} min.")

        plt.figure("fft")
        phi_fft = np.fft.rfft(angle_arr)
        phi_fft_f = np.fft.rfftfreq(len(angle_arr), d=meas_interval * 3600)

        plt.plot(3600 * phi_fft_f[1:], np.abs(phi_fft)[1:])
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

        plt.figure("Pulse shift")
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
        plt.title(f"Amplitude of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, ampl_change)
        plt.xlabel(f"Measurement time ({mt_unit})")
        if self.plot_settings.stability_plot_rel_change:
            plt.ylabel(r"$\Delta$A (%)")
        else:
            plt.ylabel(r"$\Delta$A (arb. u.)")

        plt.figure("Stability phase")
        plt.title(f"Phase of reference measurement at {selected_freq_} THz")
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

        if climate_log_file is not None or self.plot_settings.climate_file is not None:
            climate_plot_ret = self.plot_climate(climate_log_file, unit=(mt_unit, conv_factor))
            if climate_plot_ret is not None:
                climate_meas_times, climate_value_dict = climate_plot_ret
            else:
                return ret
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
                time_shift = np.round(highest_correlation[1] * meas_interval * 3600, 2)
                msg = f"Pearson r (climate quantity, pulse delay) for {k}: {max_corr_val}"
                msg += f" when shifted by {time_shift} seconds"
                logging.info(msg)

                plt.plot(shift_arr * meas_interval, r_vals, label=k)

            label_map = self.plot_settings.redp_sensor_labels
            plt.figure("Climate correlation plot")
            for k in plotted_climate_vals:
                x = np.gradient(plotted_climate_vals[k], 0.012186554258538694)
                x = plotted_climate_vals[k]
                if "0" in k:
                    y = relative_delay
                    p = np.polyfit(x, y, 1)
                    y = x * p[0] + p[1]
                    plt.plot(x, y, label=f"linear fit {label_map[k]}")
                    print(p)
                plt.scatter(x, relative_delay, label=label_map[k])
            plt.ylabel("Pulse shift (fs)")
            plt.xlabel("Temperature (°C)")

        return ret

    def plot_climate(self, log_file=None, unit=None, quantity=ClimateQuantity.Temperature):
        log_file = log_file if log_file else self.plot_settings.climate_file
        if log_file is None:
            return None

        full_log_path = self.dataset.find_climate_log_file(log_file)
        if not full_log_path:
            logging.info("No matching climate logfile found")
            return None
        else:
            logging.info(f"Using climate logfile: {full_log_path}")

        is_rp_log = False
        if "pitaya" in str(full_log_path):
            is_rp_log = True
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

        meas_time_diff = np.array([(t - t0).total_seconds() / 3600 for t in meas_time])

        if unit is None:
            if meas_time_diff.max() < 5 / 60:
                conv_factor = 3600
                mt_unit = "seconds"
            elif 5 / 60 <= meas_time_diff.max() < 0.5:
                conv_factor = 60
                mt_unit = "minutes"
            else:
                conv_factor = 1
                mt_unit = "h"
        else:
            mt_unit, conv_factor = unit

        if quantity == ClimateQuantity.Temperature:
            quant = temp
            y_label = r"$\theta$ (°C)"
            dy_label = r"$\partial \theta / \partial t_m$" + f" (°C/{mt_unit[0]})"
        else:
            quant = humidity
            y_label = "Humidity (%)"
            dy_label = fr"$\Delta$Humidity (\\%/{mt_unit[0]})"

        meas_time_diff *= conv_factor

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

    @action("Stability difference", group="Plots")
    def system_stability_diff_plot(self):
        system_stab_res_refs = self.plot_system_stability()
        x = system_stab_res_refs["meas_times"]
        y_ref = system_stab_res_refs["ref_relative_delay"]

        self.settings.pp_opt.window_opt.win_start = 11
        system_stab_res_mon_pulse0 = self.plot_system_stability(meas_set_kw="-sub-")
        xp = system_stab_res_mon_pulse0["meas_times"]
        fp = system_stab_res_mon_pulse0["ref_relative_delay"]
        y_pulse0 = np.interp(x, xp, fp)

        self.settings.pp_opt.window_opt.win_start = 27
        system_stab_res_mon_pulse1 = self.plot_system_stability(meas_set_kw="-sub-")
        xp = system_stab_res_mon_pulse1["meas_times"]
        fp = system_stab_res_mon_pulse1["ref_relative_delay"]
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


    @action("Image", group="Image plots")
    def plot_image(self, img_extent=None):
        self._update_fig_num()
        info = self.img_properties
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        self.grid_vals = self._calc_grid_vals()

        shown_grid_vals = self.grid_vals.real
        shown_grid_vals = shown_grid_vals[w0:w1, h0:h1]
        shown_grid_vals = self._exclude_pixels(shown_grid_vals)

        if self.plot_settings.log_scale:
            shown_grid_vals = np.log10(shown_grid_vals)

        fig = plt.figure(self.img_properties["fig_num"])
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.img_properties["extent"]

        cbar_min, cbar_max = self.plot_settings.cbar_lim
        if cbar_min is None:
            cbar_min = np.min(shown_grid_vals)
        if cbar_max is None:
            cbar_max = np.max(shown_grid_vals)

        if self.plot_settings.log_scale:
            self.settings.cbar_min = np.log10(cbar_min)
            self.settings.cbar_max = np.log10(cbar_max)

        axes_extent = (float(img_extent[0] - self.img_properties["dx"] / 2),
                       float(img_extent[1] + self.img_properties["dx"] / 2),
                       float(img_extent[2] - self.img_properties["dy"] / 2),
                       float(img_extent[3] + self.img_properties["dy"] / 2))
        img_ = ax.imshow(shown_grid_vals.transpose((1, 0)),
                         vmin=cbar_min, vmax=cbar_max,
                         origin="lower",
                         cmap=plt.get_cmap(self.plot_settings.color_map),
                         extent=axes_extent,
                         interpolation=self.plot_settings.pixel_interpolation.value
                         )
        if self.plot_settings.invert_x:
            ax.invert_xaxis()
        if self.plot_settings.invert_y:
            ax.invert_yaxis()

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        self._update_quantity_label()
        quantity_label = self.img_properties["quantity_label"]

        img_title_option = str(self.plot_settings.img_title)
        ax.set_title(" ".join([quantity_label, img_title_option]))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(img_, cax=cax)
        cbar.set_ticks(np.round(np.linspace(cbar_min, cbar_max, 4), 3))

        if self.plot_settings.en_cbar_label:
            cbar_label = self.selected_quantity.label + self.selected_quantity.unit
            cbar.set_label(cbar_label, rotation=270, labelpad=30)

        self.img_ax = ax
        self.img_properties["plotted_image"] = True

    @action("Plot measurement on image", group="Image plots")
    def _plot_meas_on_image(self, measurements):
        if not plt.fignum_exists(self.img_properties["fig_num"]):
            return

        plt.figure(num=self.img_properties["fig_num"])
        img_ax = self.img_ax

        meas_x_coords, meas_y_coords = [], []
        for m in measurements:
            meas_x_coords.append(m.position[0])
            meas_y_coords.append(m.position[1])

        plt_fun = img_ax.scatter

        plt_fun(meas_x_coords, meas_y_coords, color="black", linewidth=0.4)

    @action("Plot reference on image", group="Image plots")
    def plot_refs_on_image(self):
        self._plot_meas_on_image(self.measurements["refs"])

    @action("Line plot", group="Image plots")
    def plot_line(self, line_coords=None, direction=Direction.Horizontal, fig_num_=None, y_label=None, **plot_kwargs):
        if line_coords is None:
            line_coords = [0.0]
        if isinstance(line_coords, (int, float)):
            line_coords = [line_coords]

        horizontal = direction == direction.Horizontal

        if horizontal:
            fig_num = "x-slice"
            x_label = "x (mm)"
        else:
            fig_num = "y-slice"
            x_label = "y (mm)"

        if fig_num_ is None:
            fig_num += "_" + self.img_properties["quantity_label"].replace(" ", "_")
            plt.figure(fig_num)
        else:
            plt.figure(fig_num_)
        plt.title(f"Line scan ({direction.name})")
        plt.xlabel(x_label)
        if y_label is None:
            plt.ylabel(self.img_properties["quantity_label"])
        else:
            plt.ylabel(y_label)

        for line_coord in line_coords:
            if horizontal:
                measurements, coords = self.dataset.get_line(None, line_coord)
                actual_const_coord = measurements[0].position[1]
            else:
                measurements, coords = self.dataset.get_line(line_coord, None)
                actual_const_coord = measurements[0].position[0]

            logging.info("Calculating line values")
            vals = []
            for i, measurement in enumerate(measurements):
                msg =  f"{round(100 * i / len(measurements), 2)} % done. "
                msg += f"(Measurement: {i+1}/{len(measurements)}, {measurement.position} mm)"
                if i == len(measurements) - 1:
                    msg += "\n"
                logging.info(msg)

                vals.append(self.grid_func(measurement))

            if horizontal:
                if not "label" in plot_kwargs:
                    plot_kwargs["label"] = f"y={actual_const_coord} (mm)"
                plt.plot(coords, vals, **plot_kwargs)
            else:
                if not "label" in plot_kwargs:
                    plot_kwargs["label"] = f"x={actual_const_coord} (mm)"
                plt.plot(coords, vals, **plot_kwargs)

        self._plot_meas_on_image(measurements)

    def plot_q_space_eval_result(self, res):
        pass
    
    def plot_jitter(self):
        x = [25, 50, 100, 200]
        y = [113.8, 39.8, 12.47, 6.17]

        plt.figure("Jitter")
        plt.plot(x, y)
        plt.xlabel("Measurement window (ps)")
        plt.ylabel("Largest jump (fs)")

    @action("Knife edge", group="Plots")
    def knife_edge(self, x=None, y=None, coord_slice=None):
        measurements, coords = self.dataset.get_line(x, y)
        vals = np.array([self.dataset.power(meas_, self.sel_freq_range) for meas_ in measurements])

        pos_axis = coords[np.nonzero(vals)]
        vals = vals[np.nonzero(vals)]

        if coord_slice is not None:
            val_mask = (coord_slice[0] < pos_axis) * (pos_axis < coord_slice[1])
            vals = vals[val_mask]
            pos_axis = pos_axis[val_mask]

        def _model(p):
            p_max, p_offset, w, h0 = p
            vals_model_ = p_offset + 0.5 * p_max * erfc(np.sqrt(2) * (pos_axis - h0) / w)

            return vals_model_

        def _cost(p):
            return np.sum((vals - _model(p)) ** 2)

        # vertical direction
        if x is not None:
            plot_x_label = "y (mm)"
        else:  # horizontal direction
            plot_x_label = "x (mm)"

        plt.figure("Knife edge")
        plt.xlabel(plot_x_label)
        plt.ylabel(f"Power (arb. u.) summed over {self.sel_freq_range[0]}-{self.sel_freq_range[1]} THz")

        p0 = np.array([vals[0], 0.0, 0.5, 34.0])
        opt_res = shgo(_cost, bounds=([p0[0] - 1, p0[0] + 1],
                                      [p0[1], p0[1] + 0.01],
                                      [p0[2] - 0.4, p0[2] + 2.0],
                                      [p0[3] - 2, p0[3] + 2]))

        plt.scatter(pos_axis, vals, label="Measurement", s=30, c="red", zorder=3)
        plt.plot(pos_axis, _model(opt_res.x), label="Optimization result")
        plt.plot(pos_axis, _model(p0), label="Initial guess")

        s = "\n".join(["".join(s) for s in
                       zip(["$P_{max}$: ", "$P_{offset}$: ", "Beam radius: ", "$h_0$: "],
                           [str(np.round(np.abs(param), 2)) for param in opt_res.x],
                           ["", "", " (mm)", " (mm)"])])
        plt.text(pos_axis[0], 0.04, s)

        return opt_res


    def plot_eval_res(self, res):
        freq_axis = self.dataset.freq_axis

        n_sub = res["n_sub"]

        t_exp_1layer, t_exp_2layer = res["t_exp_1layer"], res["t_exp_2layer"]

        f0, f1 = self.settings.eval_opt.fit_range
        f_mask = (f0 < freq_axis) * (freq_axis < f1)

        plt.figure("TEST2")
        phi_1l = np.unwrap(np.angle(t_exp_1layer[f_mask]))
        phi_2l = np.unwrap(np.angle(t_exp_2layer[f_mask]))
        plt.plot(freq_axis[f_mask], phi_1l, label="Experiment 1l")
        plt.plot(freq_axis[f_mask], phi_2l, label="Experiment 2l")
        # self.phase_fit_(freq_axis[f_mask], phi_1l)
        # self.phase_fit_(freq_axis[f_mask], phi_2l)

        plt.figure("n_sub")
        plt.plot(freq_axis, n_sub.real, label="Real part")
        plt.plot(freq_axis, n_sub.imag, label="Imaginary part")
        # plt.ylim((-0.005, 0.020))
        plt.xlim(self.settings.eval_opt.fit_range)
        plt.xlabel("Frequency (THz)")

        t_mod_sub = res["t_mod_sub"]
        plt.figure("Transmission fit abs sub")
        plt.plot(freq_axis[f_mask], np.log10(np.abs(t_exp_1layer[f_mask])), label="Experiment")
        plt.plot(freq_axis[f_mask], np.log10(np.abs(t_mod_sub[f_mask])), label="Model")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("log10(|t|)")

        plt.figure("Transmission fit angle sub")
        plt.plot(freq_axis[f_mask], np.angle(t_exp_1layer[f_mask]), label="Experiment")
        plt.plot(freq_axis[f_mask], np.angle(t_mod_sub[f_mask]), label="Model")
        plt.xlabel("Frequency (THz)")

        plt.figure("Transmission fit abs sub diff")
        log_diff = np.log10((np.abs(t_exp_1layer[f_mask]) - np.abs(t_mod_sub[f_mask])) ** 2)
        plt.plot(freq_axis[f_mask], log_diff, label="Log squared difference")
        # plt.plot(freq_axis[f_mask], n_sub.imag[f_mask] / np.max(n_sub.imag[f_mask]), label="n_sub.imag")
        plt.xlabel("Frequency (THz)")

        t_mod_film = res["t_mod_film"]
        plt.figure("Transmission fit abs film")
        plt.plot(freq_axis[f_mask], np.abs(t_exp_2layer[f_mask]), label="Experiment", ls="dashed")
        plt.plot(freq_axis[f_mask], np.abs(t_mod_film[f_mask]), label="Model (fit)")
        plt.ylim((-0.05, 0.45))
        plt.xlabel("Frequency (THz)")
        plt.ylabel("|t|")

        plt.figure("Transmission fit angle film")
        plt.plot(freq_axis[f_mask], np.angle(t_exp_2layer[f_mask]), label="Experiment")
        plt.plot(freq_axis[f_mask], np.angle(t_mod_film[f_mask]), label="Model")
        plt.xlabel("Frequency (THz)")

        if "sigma_n_film" in res:
            sigma_n_film = res["sigma_n_film"]
            plt.figure("Conductivity(fit)")
            plt.plot(freq_axis, sigma_n_film.real, label="Real part")
            plt.plot(freq_axis, sigma_n_film.imag, label="Imaginary part")

        if "single_layer_eval" in res:
            single_layer_eval = res["single_layer_eval"]

            plt.figure("n_sub")
            # plt.plot(freq_axis, single_layer_eval["refr_idx"].imag, label="Imaginary part (1 layer eval)")

            plt.figure("absorption coefficient")
            plt.plot(freq_axis, 4 * np.pi * n_sub.imag * freq_axis / 0.03, label="fit")
            plt.plot(freq_axis, single_layer_eval["alpha"], label="single layer eval")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Absorption coefficient (1/cm)")
            plt.ylim((0, 0.012))
            plt.xlim((-0.05, 3.5))

        if "n_film" in res:
            n_film = res["n_film"]
            plt.figure("n_film")
            plt.plot(freq_axis, n_film.real, label="real")
            plt.plot(freq_axis, n_film.imag, label="imag")
            plt.xlabel("Frequency (THz)")

        if "sigma_exp" in res and "sigma_mod" in res:
            sigma_exp, sigma_mod = res["sigma_exp"], res["sigma_mod"]
            plt.figure("Conductivity")
            plt.plot(freq_axis[f_mask], sigma_exp[f_mask].real, label="Exp (real)")
            plt.plot(freq_axis[f_mask], sigma_exp[f_mask].imag, label="Exp (imag)")
            plt.plot(freq_axis[f_mask], sigma_mod[f_mask].real, label="Fit (real)")
            plt.plot(freq_axis[f_mask], sigma_mod[f_mask].imag, label="Fit (imag)")
            # plt.xlim((0.20, 2.55))
            # plt.ylim((-10, 200))
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Conductivity (S/cm)")

    def plot_freq_fit(self, res):
        for quantity in res:
            pass
        return

    def save_fig(self, fig_num_, filename=None, **kwargs):
        save_dir = Path(self.settings.export_csv_dir)
        filetype = self.settings.save_settings.filetype
        kwargs.setdefault("dpi", self.settings.save_settings.dpi)
        kwargs.setdefault("bbox_inches", self.settings.save_settings.bbox_inches)
        kwargs.setdefault("pad_inches", self.settings.save_settings.pad_inches)

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

        w = self.settings.save_settings.set_size_inches
        fig.set_size_inches(w=w, forward=False)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(save_dir / (filename_s + f".{filetype}"), **kwargs)

    @action("Close all figures", group="Show / close plots")
    def close_figures(self):
        for fig_num in plt.get_fignums():
            plt.close(fig_num)

    @action("Show plots", group="Show / close plots")
    def plt_show(self, save_file_suffix=None, only_save_plots=False):

        # fig_labels = [plt.figure(fig_num).get_label() for fig_num in plt.get_fignums()]
        only_shown_fig_nums = []
        if self.plot_settings.only_shown_figures:
            only_shown_fig_nums = self.plot_settings.only_shown_figures
            logging.warning(f"Only showing figures {self.plot_settings.only_shown_figures}")

        if self.plot_settings.disable_legend:
            figs_w_disabled_legends = self.plot_settings.disable_legend
            logging.warning(f"Legends disabled for figure: {figs_w_disabled_legends}")

        not_shown = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig_label = fig.get_label()
            axes = fig.get_axes()
            if not any([ax.get_legend() for ax in axes]):
                for ax in axes:
                    h, labels = ax.get_legend_handles_labels()
                    if labels and not (fig_label in self.plot_settings.disable_legend):
                        ax.legend()

            if self.settings.save_settings.save_plots:
                save_file=None
                if save_file_suffix:
                    save_file = str(fig_num) + "_" + save_file_suffix
                self.save_fig(fig_num, filename=save_file)

            if only_save_plots:
                plt.close(fig_num)
                continue

            if only_shown_fig_nums and fig_label not in only_shown_fig_nums:
                not_shown.append(fig_label)
                plt.close(fig_num)
                continue

            shown_plots_dict = self.plot_settings.traits(group="Shown plots")
            for shown_plot_num in shown_plots_dict:
                if (shown_plot_num in fig_label) and (not shown_plots_dict[shown_plot_num]):
                    not_shown.append(fig_label)
                    plt.close(fig_num)
            """
            if fig_label in self.settings.shown_plots:
                if not self.settings.shown_plots[fig_label]:
                    not_shown.append(fig_label)
                    plt.close(fig_num)"""

        logging.info(f"Not showing plots: {', '.join(not_shown)}")
        plt.show()




import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np
from traitlets import Bool, Float, Enum as TEnum, observe
from common.components import action
from common.consts import c_thz, eps0_thz
from common.default_appsettings import QuantityEnum
from common.measurements import Measurement
from common.traits import Q_, Quantity, ValueRange, Path as TPath
from common.eval_component.conductivity_models import sigma_to_n
from common.eval_component.eval_component_base import EvalComponentBase
from common.eval_component.eval_result import EvalResultData, SingleResultData
from common.eval_component.quantity_set import QuantityDataSet
from common.eval_component.transfer_functions import TransmissionModels

"""
Conductivity of a thin film on a substrate. All calculations use the e^{-iωt} (physics) convention, the measured
transmission (dataset.transmission) is its complex conjugate.
Units: [freq] = THz, [d] = µm, [sigma] = S/cm, [time shift] = fs
"""


class SubstrateIndexSource(Enum):
    thick_sample = "Thick sample estimate (substrate point)"
    qspace_result = "Q-space fit result"
    constant = "Constant"


class ConductivityMethod(Enum):
    tinkham = "Thin film approximation (substrate reference)"
    two_layer = "Two-layer model (air reference)"


two_layer_models = [m for m in TransmissionModels if getattr(m.value, "layer_cnt", 1) == 2]


def tinkham_conductivity(t_film, t_sub, n_sub, d_film):
    # thin film approximation, t_film and t_sub relative to the same reference:
    # t_film / t_sub = (1 + n_sub) / (1 + n_sub + Z0 sigma d_film), 1 / Z0 = eps0 c, 1e4: S/µm -> S/cm
    return 1e4 * eps0_thz * c_thz * (1 + n_sub) / d_film * (t_sub / t_film - 1)


def solve_two_layer_conductivity(t_film, freq, n_sub, d_sub, d_film, eps_inf, model, model_kwargs,
                                 max_iter=50, rel_tol=1e-10):
    # solves model(sigma) = t_film for the complex sigma at each frequency (and measurement) with a damped complex
    # Newton iteration, starting from the thin film approximation. t_film: physics convention, shape (..., F)
    kwargs = {**model_kwargs, "d": d_sub, "h": d_film, "n_sub": n_sub, "shift": 0}

    def t_model(sigma):
        return np.conj(model(sigma_to_n(freq, sigma, eps_inf), freq, **kwargs))

    t_sub_model = t_model(np.zeros_like(t_film))  # film without free carriers
    sigma = tinkham_conductivity(t_film, t_sub_model, n_sub, d_film)

    residual = t_model(sigma) - t_film
    tol = rel_tol * np.abs(t_film)
    for _ in range(max_iter):
        if np.all(np.abs(residual) <= tol):
            break
        h = 1e-6 * np.maximum(np.abs(sigma), 1.0)
        derivative = (t_model(sigma + h) - (residual + t_film)) / h
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.nan_to_num(residual / derivative)

        damping = np.ones(sigma.shape)
        for _ in range(10):
            trial = sigma - damping * step
            trial_residual = t_model(trial) - t_film
            worse = np.abs(trial_residual) > np.abs(residual)
            if not worse.any():
                break
            damping = np.where(worse, damping / 2, damping)

        improved = np.abs(trial_residual) <= np.abs(residual)
        if not improved.any():
            break
        sigma = np.where(improved, trial, sigma)
        residual = np.where(improved, trial_residual, residual)

    converged = np.abs(residual) <= np.maximum(tol, 1e-8)

    return sigma, t_model(sigma), converged


class ConductivityEval(EvalComponentBase):
    # not used here
    transmission_model = None
    selected_cost_fun = None
    number_of_workers = None

    film_grp = "Thin film"
    d_film = Quantity(Q_(0.1, "µm"), group=film_grp).tag(name="Film thickness", priority=0)
    eps_inf_film = Float(1.0, group=film_grp,
                         help="Permittivity of the film without the conductivity contribution, "
                              "eps = eps_inf + i sigma / (eps0 omega)").tag(name="Film background permittivity",
                                                                         priority=1)
    film_time_shift = Quantity(Q_(0.0, "fs"), group=film_grp,
                               help="Positive if the pulse through the film sample is delayed by this time compared "
                                    "to the substrate (e.g. a thicker substrate), it is removed before the "
                                    "evaluation").tag(name="Film pulse time shift", priority=2)

    sub_grp = "Substrate"
    sub_pnt = ValueRange([Q_(0.0, "mm"), Q_(0.0, "mm")], group=sub_grp,
                         help="Position of the bare substrate measurement(s)").tag(name="Substrate point (x, y)",
                                                                                   priority=0)
    sub_index_source = TEnum(SubstrateIndexSource, default_value=SubstrateIndexSource.thick_sample,
                             group=sub_grp).tag(name="Substrate refractive index", priority=1)
    sub_result_path = TPath(Path("."), must_exist=False, group=sub_grp,
                            help="Transmission fit (QSpaceEval) result of the bare substrate. The optimum of the "
                                 "average (or first) measurement is used").tag(name="Q-space result file",
                                                                               priority=2)
    use_result_thickness = Bool(True, group=sub_grp).tag(name="Use substrate thickness from result", priority=3)
    d_sub = Quantity(Q_(500.0, "µm"), group=sub_grp).tag(name="Substrate thickness", priority=4)
    n_sub_const = ValueRange([3.4, 0.0], group=sub_grp).tag(name="Substrate n, k", priority=5)

    method_grp = "Conductivity evaluation"
    method = TEnum(ConductivityMethod, default_value=ConductivityMethod.tinkham, group=method_grp,
                   help="Thin film approximation: sigma from the ratio of the substrate and film transmission. "
                        "Two-layer model: sigma solved at each frequency from the film transmission with the "
                        "selected two-layer model, needs the substrate refractive index and thickness"
                   ).tag(name="Method", priority=0)
    two_layer_model = TEnum(two_layer_models, default_value=TransmissionModels.tmm_2layer_no_sub_echo,
                            group=method_grp,
                            help="tmm_2layer_no_sub_echo: no substrate echoes (windowed / short traces), "
                                 "tmm_2layer and model_2layer: with all substrate echoes, t_model_2layer: "
                                 "number of echoes from the Fabry-Perot count setting").tag(name="Two-layer model",
                                                                                           priority=1)

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset.register_quantity_func(QuantityEnum.Conductivity, self.conductivity_quantity)

    @property
    def _ui_control_widget(self):
        return getattr(self, "_ui_widget_internal", None)

    @_ui_control_widget.setter
    def _ui_control_widget(self, value):
        self._ui_widget_internal = value
        if value:
            self._update_visibility()

    @observe("method", "sub_index_source", "use_result_thickness")
    def _update_visibility(self, change=None):
        source, is_tinkham = self.sub_index_source, self.method == ConductivityMethod.tinkham
        visible = {
            "sub_pnt": is_tinkham or source == SubstrateIndexSource.thick_sample,
            "sub_index_source": True,
            "sub_result_path": source == SubstrateIndexSource.qspace_result,
            "use_result_thickness": source == SubstrateIndexSource.qspace_result,
            "d_sub": not (source == SubstrateIndexSource.qspace_result and self.use_result_thickness),
            "n_sub_const": source == SubstrateIndexSource.constant,
        }
        self.toggle_traits([k for k, v in visible.items() if v], group_filter=self.sub_grp)
        self.toggle_traits(["method"] + ["two_layer_model"] * (not is_tinkham), group_filter=self.method_grp)

    def substrate_measurements(self):
        meas_list = self.measurement_selector.get_measurements_from_point(*self.sub_pnt)
        meas_list = list(meas_list) if isinstance(meas_list, (list, tuple)) else [meas_list]
        sub_meas = [m for m in meas_list if not self.dataset.is_ref(m)]
        if not sub_meas:
            raise ValueError(f"No sample measurement at the substrate point {self.sub_pnt}")

        return sub_meas

    def load_substrate_result(self, freq_axis):
        path = Path(self.sub_result_path)
        if not path.is_file():
            raise ValueError(f"Substrate result file '{path}' not found")

        result_data = self.current_result.parse_hdf5(path)
        optimum = result_data.optimum_results
        key = optimum.get("Average") or next(iter(optimum.values()), None) or next(iter(result_data.results), None)
        if key is None:
            raise ValueError(f"No results in '{path}'")
        result = result_data.results[key]

        datasets = result.datasets
        if "n_real" in datasets and "n_imag" in datasets:
            n_data = datasets["n_real"].data.magnitude + 1j * datasets["n_imag"].data.magnitude
            f_res = datasets["n_real"].axes[0].to("THz").magnitude
        elif "n" in datasets:
            n_data = datasets["n"].data.magnitude
            f_res = datasets["n"].axes[0].to("THz").magnitude
        else:
            raise ValueError(f"Result '{key}' in '{path}' contains no refractive index")

        if freq_axis[0] < f_res[0] or freq_axis[-1] > f_res[-1]:
            logging.warning(f"Substrate result covers {f_res[0]:.2f}-{f_res[-1]:.2f} THz only, the refractive index "
                            f"is extended with its end values outside of this range")
        n_sub = np.interp(freq_axis, f_res, n_data.real) + 1j * np.interp(freq_axis, f_res, n_data.imag)
        logging.info(f"Substrate refractive index from {path.name} ({result.measurement}, d = {result.d}, "
                     f"model {result_data.model_name})")

        # the substrate index is only consistent with a two-layer model that treats the substrate echoes the same way
        with_echoes = {"tmm_1layer", "model_1layer_inf"}
        if self.method == ConductivityMethod.two_layer:
            fit_has_echoes = result_data.model_name in with_echoes
            model_has_echoes = self.two_layer_model in (TransmissionModels.tmm_2layer, TransmissionModels.model_2layer)
            if fit_has_echoes != model_has_echoes:
                logging.warning(f"The substrate was fitted with {result_data.model_name} "
                                f"({'with' if fit_has_echoes else 'without'} substrate echoes) but the two-layer model "
                                f"{self.two_layer_model.name} is {'with' if model_has_echoes else 'without'} them, "
                                f"this usually shows up as oscillations in the conductivity")

        return n_sub, result.d.to("µm").magnitude

    def substrate_properties(self, freq_idx):
        # n_sub, d_sub and (if needed) the measured substrate transmission (physics convention) on the given indices
        freq_axis = self.dataset.freq_axis[freq_idx]
        d_sub = self.d_sub.to("µm").magnitude
        source = self.sub_index_source

        t_sub, sub_meas = None, None
        if self.method == ConductivityMethod.tinkham or source == SubstrateIndexSource.thick_sample:
            sub_meas = self.substrate_measurements()
            sub_refs = self.measurement_selector.get_matching_refs(sub_meas)
            t_sub = np.conj(np.mean(self.dataset.transmission(sub_meas, sub_refs), axis=0)[freq_idx])

        match source:
            case SubstrateIndexSource.thick_sample:
                window_res = self.dataset.windowing_eval(sub_meas, d=d_sub)
                n_sub = np.mean(window_res.refr_idx, axis=0)[freq_idx]
            case SubstrateIndexSource.qspace_result:
                n_sub, d_result = self.load_substrate_result(freq_axis)
                if self.use_result_thickness:
                    d_sub = d_result
            case _:
                n, k = self.n_sub_const
                n_sub = (n + 1j * k) * np.ones_like(freq_axis, dtype=complex)

        return n_sub, d_sub, t_sub

    def calculate(self, meas_list, freq_idx):
        # conductivity of the film for each measurement, arrays (N_meas, F) on the given frequency indices
        freq_axis = self.dataset.freq_axis[freq_idx]
        n_sub, d_sub, t_sub = self.substrate_properties(freq_idx)
        d_film = self.d_film.to("µm").magnitude
        if d_film <= 0:
            raise ValueError("Film thickness must be larger than 0")

        refs = self.measurement_selector.get_matching_refs(meas_list)
        t_meas = self.dataset.transmission(meas_list, refs)[:, freq_idx]
        dt = self.film_time_shift.to("ps").magnitude
        t_film = np.conj(t_meas * np.exp(1j * 2 * np.pi * freq_axis * dt))

        res = {"n_sub": n_sub, "d_sub": d_sub, "t_film": t_film, "t_sub": t_sub}
        if self.method == ConductivityMethod.tinkham:
            res["sigma"] = tinkham_conductivity(t_film, t_sub, n_sub, d_film)
            res["t_mod"], res["converged"] = None, np.ones(t_film.shape, dtype=bool)
        else:
            model_kwargs = {"n1": 1, "n4": 1, "nfp": self.settings.eval_opt.fp_count}
            res["sigma"], res["t_mod"], res["converged"] = solve_two_layer_conductivity(
                t_film, freq_axis, n_sub, d_sub, d_film, self.eps_inf_film, self.two_layer_model.value,
                model_kwargs)

        res["n_film"] = sigma_to_n(freq_axis, res["sigma"], self.eps_inf_film)

        return res

    def conductivity_quantity(self, meas_list):
        # QuantityEnum.Conductivity (image plot, regression): (N_meas, F) on the full frequency axis, nan outside
        # of the evaluation frequency range
        meas_list = list(meas_list)
        sigma = np.full((len(meas_list), len(self.dataset.freq_axis)), np.nan, dtype=complex)
        try:
            sigma[:, self.freq_range_idx] = self.calculate(meas_list, self.freq_range_idx)["sigma"]
        except ValueError as e:
            logging.error(f"Conductivity: {e}")

        return sigma

    @action("Evaluate conductivity", group=method_grp, priority=10)
    def evaluate_conductivity(self):
        if self._active_eval is not None:
            logging.warning("Conductivity evaluation is already running")
            return
        selection = self.measurement_selector.get_selection()
        if selection is None or not selection.sams:
            logging.warning("No film measurements selected")
            return
        meas_list = list(selection.sams)

        evaluator = self
        self._active_eval = evaluator

        def bg_worker():
            try:
                res = self.calculate(meas_list, self.freq_range_idx)
                self._result_ready(self.prepare_result(meas_list, res))
            except ValueError as e:
                logging.error(f"Conductivity: {e}")
            except Exception:
                traceback.print_exc()
            finally:
                if self._active_eval is evaluator:
                    self._active_eval = None

        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._thread_executor.submit(bg_worker)

    def prepare_result(self, meas_list, res) -> EvalResultData:
        freq_axis = Q_(self.freq_axis, "THz")
        is_tinkham = self.method == ConductivityMethod.tinkham

        eval_result_data = EvalResultData()
        eval_result_data.result_type = "Conductivity"
        eval_result_data.dataset_path = self.dataset.data_path
        eval_result_data.model_name = self.method.name if is_tinkham else self.two_layer_model.name
        eval_result_data.measurement_quantity = QuantityEnum.Conductivity.name

        per_meas = {"sigma": res["sigma"], "n_film": res["n_film"], "t_film": res["t_film"]}
        if res["t_mod"] is not None:
            per_meas["t_mod"] = res["t_mod"]
        formatted = {key: self.format_meas_dict(meas_list, data) for key, data in per_meas.items()}

        not_converged = int(np.sum(~res["converged"]))
        if not_converged:
            logging.warning(f"Conductivity did not converge at {not_converged} frequency point(s)")

        for res_idx, meas in enumerate(formatted["sigma"]):
            label = meas.filepath.name if isinstance(meas, Measurement) else str(meas)
            single = SingleResultData(measurement=label, d=self.d_film.to("µm"), shift=self.film_time_shift,
                                      freq_axis=freq_axis)
            single.optimization_info = {
                "method": self.method.value,
                "model": "" if is_tinkham else self.two_layer_model.name,
                "substrate_index_source": self.sub_index_source.value,
                "d_sub": Q_(res["d_sub"], "µm"),
                "eps_inf_film": self.eps_inf_film,
                "converged": not_converged == 0,
                "not_converged_points": not_converged,
                "timestamp": str(datetime.now().isoformat()),
            }

            units = {"sigma": "S/cm", "n_film": "", "t_film": "", "t_mod": ""}
            labels = {"sigma": "Conductivity", "n_film": "Film refractive index",
                      "t_film": "Film transmission (measured)", "t_mod": "Film transmission (model)"}
            for key, meas_data in formatted.items():
                data = meas_data[meas]
                self._add_dataset(single.datasets, key, data[:, 1], data[:, 2], units[key], labels[key],
                                  polar=key.startswith("t_"))
            self._add_dataset(single.datasets, "n_sub", res["n_sub"], None, "", "Substrate refractive index")

            key = f"result_{res_idx:03d}"
            eval_result_data.results[key] = single
            eval_result_data.optimum_results[label] = key
            eval_result_data.measurement_names.append(label)

        return eval_result_data

    def _add_dataset(self, datasets, key, data, uncert, unit, label, polar=False):
        # complex data is split into real / imaginary (or magnitude / phase) parts for plotting and saving
        freq_axis = Q_(self.freq_axis, "THz")
        uncert = np.zeros_like(data) if uncert is None else uncert
        if polar:  # uncert: std of real and imaginary part, |std| as magnitude uncertainty, |std| / |data| for phase
            with np.errstate(divide="ignore", invalid="ignore"):
                phase_uncert = np.nan_to_num(np.abs(uncert) / np.abs(data))
            parts = {"abs": (np.abs(data), np.abs(uncert), unit, "Magnitude"),
                     "phi": (np.angle(data), phase_uncert, "rad", "Phase")}
        else:
            parts = {"real": (np.real(data), np.abs(np.real(uncert)), unit, "Real"),
                     "imag": (np.imag(data), np.abs(np.imag(uncert)), unit, "Imag")}
        for part, (part_data, part_uncert, part_unit, part_label) in parts.items():
            datasets[f"{key}_{part}"] = QuantityDataSet(axes=[freq_axis],
                                                        data=Q_(part_data, part_unit),
                                                        uncert=Q_(part_uncert, part_unit),
                                                        axes_labels=["Frequency"],
                                                        data_label=f"{label} ({part_label})")

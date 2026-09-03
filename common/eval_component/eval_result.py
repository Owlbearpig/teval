from pathlib import Path

import h5py
import numpy as np
from PySide6.QtCore import QObject, Signal
from common.components import ComponentBase, action
from common.eval_component.conductivity_models import model_params
from common.eval_component.quantity_set import DataSet as SingleQuantityDataSet
from common.eval_component.quantity_set import DataSetDict as QuantityDictClass, DataSet
from common.measurements import Measurement
from common.traits import QuantityDict, Path as TPath, Quantity, Q_, TList
from traitlets import Bool, Float, Unicode, Integer, observe, Dict

# testing
p = Path(r"/media/storage/ArchivedData/Conductivity/Furtwangen/Vanadium Oxide/img0")
p = p / r"2025-01-30T18-06-22.711457-20avg-ref-X_15.000 mm-Y_-10.000 mm.txt"
from itertools import product
from datetime import datetime
# testing

class ResultSignal(QObject):
    received_result = Signal(dict)
    result_ready = Signal(object)


class EvalResult(ComponentBase):
    quantity_dict = QuantityDict()

    t_fit_res_grp_name = "Transmission fit result values"
    d = Quantity(Q_(0, "µm"), read_only=True, group=t_fit_res_grp_name)
    q_val = Quantity(Q_(0.0, ""), read_only=True, group=t_fit_res_grp_name)
    gof = Quantity(Q_(0.0, ""), read_only=True, group=t_fit_res_grp_name)
    shift = Quantity(Q_(0.0, "fs"), read_only=True, group=t_fit_res_grp_name)

    reg_result_grp_name = "Regression result values"
    fun = Float(0.0, read_only=True, group=reg_result_grp_name)
    nit = Integer(0, read_only=True, group=reg_result_grp_name)
    sig0 = Quantity(Q_(0, "S/cm"), read_only=True, group=reg_result_grp_name).tag(name="σ₀")
    tau = Quantity(Q_(0, "fs"), read_only=True, group=reg_result_grp_name).tag(name="τ")
    wp = Quantity(Q_(0, "THz"), read_only=True, group=reg_result_grp_name).tag(name="ωₚ")
    eps_inf = Float(0, read_only=True, group=reg_result_grp_name).tag(name="ε_inf")
    eps_s = Float(0, read_only=True, group=reg_result_grp_name).tag(name="ε_s")
    c1 = Float(0, read_only=True, group=reg_result_grp_name).tag(name="c₁")

    measurement = Unicode("", read_only=True).tag(priority=0, name="Measurement")
    result_type = Unicode("None", read_only=True).tag(priority=1, name="Result type")
    model_name = Unicode("", read_only=True).tag(priority=2, name="Model")
    timestamp = Unicode("", read_only=True).tag(priority=3, name="Timestamp")
    dataset_path = TPath(Path("."), read_only=True).tag(priority=4, name="Dataset path")
    sub_dataset_path = TPath(Path("."), read_only=True).tag(priority=5, name="Sub. dataset path")
    converged = Bool(False, read_only=True).tag(priority=6, name="Converged")

    measurement_list = TList(group="test", read_only=True)
    thicknesses = TList(group="thicknesses", read_only=True)
    shifts = TList(group="shifts", read_only=True)

    opt_res_dict = Dict()

    @action(name="test")
    def test(self):

        def opt_res(task, m):
            d, shift = task
            freq_axis = Q_(np.linspace(0.5, 2, 4000), "THz")
            ret = {"d": Q_(d, "µm"),
                   "shift": Q_(shift, "fs"),
                   "q_val": Q_(np.random.random(), ""),
                   "gof": Q_(np.random.random(), ""),
                   "converged": True,

                   # Strings
                   "timestamp": str(datetime.now().isoformat()),
                   "measurement": str(m),

                   # Datasets ( Q_(x) )
                   "n0": DataSet(axes=[freq_axis],
                                 data=Q_(np.random.random(freq_axis.shape[0]), ""),
                                 data_label="Simple n",
                                 axes_labels=["Frequency"]),
                   }
            return ret

        thicknesses = [100, 200, 300, 400, 500, 600]
        shifts = [-0.5, 0, 1.5]
        meas_list = ["Average", Measurement(p)]
        all_measurement_results = {
            "result_type": "Transmission fit",
            "measurements": meas_list,
            "model_name": "tmm_1layer",
            "measurement_quantity": "Transmission",
            "optimization_results": {},
        }
        for meas in meas_list:
            tasks = product(thicknesses, shifts)
            parsed_opt_res_dict = {task: opt_res(task, meas) for task in tasks}
            all_measurement_results["optimization_results"][meas] = parsed_opt_res_dict

        self.parse_opt_res_dict(all_measurement_results, is_loading=True)

    @action(name="Show Q-space plot")
    def plot_q_space(self):
        pass

    """
    @observe("measurement_list", "thicknesses", "shifts")
    def on_selection_change(self, change): 
        print(change)
    """

    @observe("measurement_list")
    def on_selection_change(self, change):
        print(change)
        meas_opt_results = self.opt_res_dict["optimization_results"][change["new"]]

        self.set_trait("thicknesses", )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result_carrier = ResultSignal()
        self.result_carrier.received_result.connect(self.parse_opt_res_dict)

    def load_result(self, res_path):
        res_dict = {}
        if res_path.suffix == ".npz":
            res_dict = self.parse_npz(res_path)
        elif res_path.suffix == ".hdf5":
            res_dict = self.parse_hdf5(res_path)

        self.parse_opt_res_dict(res_dict, is_loading=True)

    def parse_hdf5(self, res_path):
        with h5py.File(res_path, "r") as f:
            parsed_result_dict = {}

            if "scalars" in f:
                for k in f["scalars"].keys():
                    dset = f["scalars"][k]
                    val = dset[()]

                    if isinstance(val, bytes):
                        val = val.decode("utf-8")

                    if "unit" in dset.attrs:
                        unit_str = dset.attrs["unit"]
                        if isinstance(unit_str, bytes):
                            unit_str = unit_str.decode("utf-8")

                        val = Q_(val, unit_str)
                    if "path_type" in dset.attrs:
                        val = Path(val)

                    parsed_result_dict[k] = val

            if "quantity_dict" in f:
                qd_group = f["quantity_dict"]

                for k in qd_group.keys():
                    dataset_group = qd_group[k]

                    data_dset = dataset_group["data"]

                    d_unit = data_dset.attrs["unit"]
                    data_label = data_dset.attrs["data_label"]
                    data_q = Q_(data_dset[()], d_unit)

                    axes_q, axes_labels = [], []
                    axes_group = dataset_group["axes"]

                    i = 0
                    while f"axis_{i}" in axes_group:
                        ax_subgroup = axes_group[f"axis_{i}"]
                        axis_dset = ax_subgroup["axis_dset"]

                        ax_unit = axis_dset.attrs["unit"]
                        axes_labels.append(axis_dset.attrs["axis_label"])
                        axes_q.append(Q_(axis_dset[()], ax_unit))
                        i += 1

                    parsed_result_dict[k] = DataSet(data=data_q, axes=axes_q,
                                                    data_label=data_label, axes_labels=axes_labels)

        return parsed_result_dict

    def parse_npz(self, path):
        def assemble_dataset(prefix_, npz_dict_):
            data_unit, data_magnitude = None, None
            axes_magnitude_idx_tuples, axes_unit_idx_tuples = [], []
            for k, v in npz_dict_.items():
                if f"{prefix_}__DSK__axes_magnitude" == "_".join(k.split("_")[:-1]):
                    idx_ = int(k.split("_")[-1])
                    axes_magnitude_idx_tuples.append((v, idx_))
                elif f"{prefix_}__DSK__axes_units" == "_".join(k.split("_")[:-1]):
                    idx_ = int(k.split("_")[-1])
                    axes_unit_idx_tuples.append((v.item(), idx_))
                elif k == f"{prefix_}__DSK__data_magnitude":
                    data_magnitude = v
                elif k == f"{prefix_}__DSK__data_units":
                    data_unit = v.item()

            axes_magnitudes = [t[0] for t in sorted(axes_magnitude_idx_tuples, key=lambda x: x[1])]
            axes_units = [t[0] for t in sorted(axes_unit_idx_tuples, key=lambda x: x[1])]

            axes = [Q_(*z) for z in zip(axes_magnitudes, axes_units)]
            data = Q_(data_magnitude, data_unit)

            return SingleQuantityDataSet(data, axes)

        npz_dict = dict(np.load(path, allow_pickle=False))

        parsed_result_dict = {}
        for k, v in npz_dict.items():
            if "__data_magnitude" in k[-len("__data_magnitude"):]:
                prefix = k.split("__DSK__data_magnitude")[0]
                parsed_result_dict[prefix] = assemble_dataset(prefix, npz_dict)
            elif "__QK__quantity_magnitude" in k:
                prefix = k.split("__QK__quantity_magnitude")[0]
                unit = npz_dict[f"{prefix}__QK__quantity_units"]
                parsed_result_dict[prefix] = Q_(v.item(), unit.item())
            elif ("QK" not in k) and ("DSK" not in k):
                parsed_result_dict[k] = v.item()

        return parsed_result_dict

    def parse_opt_res_dict(self, opt_res_dict, is_loading=False):
        if not opt_res_dict:
            return
        self.opt_res_dict = opt_res_dict
        self.set_trait("measurement_list", opt_res_dict["measurements"])

        self.set_trait("thicknesses", )
        thicknesses = TList(group="thicknesses", read_only=True)
        shifts = TList(group="shifts", read_only=True)

        # print(opt_res_dict)
        return
        for k, v in opt_res_dict.items():
            if isinstance(v, (int, str, float, Q_, Path)):
                self.set_trait(k, v)

        dataset_dict = {k: v for k, v in opt_res_dict.items() if isinstance(v, DataSet)}
        self.quantity_dict = QuantityDictClass(dataset_dict)

        if opt_res_dict["result_type"] == "Regression":
            active_parameters = model_params(opt_res_dict["model_name"])
            self.toggle_traits(active_parameters, group_filter=self.reg_result_grp_name)
            self.toggle_traits([], group_filter=self.t_fit_res_grp_name)
        elif opt_res_dict["result_type"] == "Transmission fit":
            self.toggle_traits([], group_filter=self.reg_result_grp_name)
            self.toggle_traits(self.traits(group=self.t_fit_res_grp_name), group_filter=self.t_fit_res_grp_name)

        if not is_loading:
            self.result_carrier.result_ready.emit(self)

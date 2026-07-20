from common.components import ComponentBase
from common.traits import QuantityDict
from common.eval_component.quantity_set import DataSetDict as QuantityDictClass, DataSet
from common.eval_component.conductivity_models import model_params
from traitlets import Bool, Float, Int, Unicode, Integer
from common.traits import Quantity, Q_
import numpy as np
from common.eval_component.quantity_set import DataSet as SingleQuantityDataSet
import h5py
import inspect


class EvalResult(ComponentBase):

    quantity_dict = QuantityDict()

    d = Quantity(Q_(0, "µm"), read_only=True, group="Transmission fit result values")
    q_val = Quantity(Q_(0, ""), read_only=True, group="Transmission fit result values")
    gof = Quantity(Q_(0, ""), read_only=True, group="Transmission fit result values")
    shift = Quantity(Q_(0, "fs"), read_only=True, group="Transmission fit result values")

    reg_res_grp_name = "Regression result values"
    fun = Float(0, read_only=True, group=reg_res_grp_name)
    nit = Integer(0, read_only=True, group=reg_res_grp_name)
    sig0 = Quantity(Q_(0, "S/cm"), read_only=True, group=reg_res_grp_name).tag(name="σ₀")
    tau = Quantity(Q_(0, "fs"), read_only=True, group=reg_res_grp_name).tag(name="τ")
    wp = Quantity(Q_(0, "THz"), read_only=True, group=reg_res_grp_name).tag(name="ωₚ")
    eps_inf = Float(0, read_only=True, group=reg_res_grp_name).tag(name="ε_inf")
    eps_s = Float(0, read_only=True, group=reg_res_grp_name).tag(name="ε_s")
    c1 = Float(0, read_only=True, group=reg_res_grp_name).tag(name="c₁")

    result_type = Unicode("None", read_only=True).tag(priority=1)
    timestamp = Unicode("", read_only=True)
    converged = Bool(False, read_only=True)
    model_name = Unicode("", read_only=True)
    measurement_quantity = Unicode("", read_only=True)

    def __init__(self, opt_res_dict=None, **kwargs):
        super().__init__(**kwargs)
        if opt_res_dict is None:
            return

        self.set_traits_from_dict(opt_res_dict)

    def load_result(self, res_path):
        res_dict = {}
        if res_path.suffix == ".npz":
            res_dict = self.parse_npz(res_path)
        elif res_path.suffix == ".hdf5":
            res_dict = self.parse_hdf5(res_path)
        self.set_traits_from_dict(res_dict)

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

    def set_traits_from_dict(self, trait_value_dict):
        for k, v in trait_value_dict.items():
            if isinstance(v, (int, str, float, Q_)):
                self.set_trait(k, v)

        dataset_dict = {k: v for k, v in trait_value_dict.items() if isinstance(v, DataSet)}
        self.quantity_dict = QuantityDictClass(dataset_dict)

        if "model_name" in trait_value_dict:
            param_keys = model_params(trait_value_dict["model_name"])
            self.toggle_traits(param_keys, self, group_filter=self.reg_res_grp_name)

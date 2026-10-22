import logging
import re
from enum import Enum
from pathlib import Path

import numpy as np
from traitlets import Enum as TEnum, Unicode, List, observe

from common.components import ComponentBase, action
from common.consts import c_thz
from common.eval_component.eval_result import EvalResult
from common.eval_component.transfer_functions import layer_n
from common.traits import Quantity, Q_, ValueRange, Path as TPath, StrListSelection, StrList


class LayerType(Enum):
    unknown = "Unknown (fitted)"
    result = "Known, from result file"
    constant = "Known, constant n"


class WaveDirection(Enum):
    top_to_bottom = "Top to bottom (enters at the first layer)"
    bottom_to_top = "Bottom to top (enters at the last layer)"


def load_layer_result(path):
    """Refractive index and thickness of the optimum result in a saved evaluation result file."""
    res_data = EvalResult.parse_hdf5(None, path)
    if not res_data.results:
        raise ValueError(f"No results in {path}")

    if res_data.optimum_results:
        meas = "Average" if "Average" in res_data.optimum_results else next(iter(res_data.optimum_results))
        res = res_data.results[res_data.optimum_results[meas]]
    elif len(res_data.results) == 1:
        res = next(iter(res_data.results.values()))
    else:
        raise ValueError(f"{path} has several results but no optimum")

    datasets = res.datasets
    if "n_real" in datasets and "n_imag" in datasets:
        freq = datasets["n_real"].axes[0].to("THz").magnitude
        n = datasets["n_real"].data.magnitude + 1j * datasets["n_imag"].data.magnitude
    elif "n" in datasets:
        freq = datasets["n"].axes[0].to("THz").magnitude
        n = np.asarray(datasets["n"].data.magnitude, dtype=complex)
    else:
        raise ValueError(f"No refractive index in {path}")

    d = res.d.to("µm").magnitude if res.d is not None else np.nan

    return np.asarray(freq, dtype=float), np.asarray(n, dtype=complex), d, res.measurement


class LayerStack(ComponentBase):
    stack_grp = "Layer stack"
    wave_direction = TEnum(WaveDirection, WaveDirection.top_to_bottom,
                           help="Direction of the THz wave through the layers, relative to the list. "
                                "With the same medium (air) on both sides the transmission is the same in "
                                "both directions").tag(name="Wave direction", group=stack_grp, priority=0)
    layer_list = StrListSelection(read_only=True).tag(group=stack_grp, priority=1, fullwidth=True)
    stack_summary = Unicode("", read_only=True).tag(name="Propagation order", group=stack_grp, priority=2,
                                                    en_save=False)
    layers = List().tag(group=stack_grp)

    editor_grp = "Layer editor"
    layer_type = TEnum(LayerType, LayerType.result).tag(name="Layer type", group=editor_grp, priority=10,
                                                         en_save=False)
    layer_label = Unicode("", help="Optional name shown in the layer list").tag(name="Label", group=editor_grp,
                                                                                priority=11, en_save=False)
    result_path = TPath(Path(""),
                        help="Saved evaluation result (HDF5) of the layer material, its optimum result "
                             "gives n and k").tag(name="Result file", group=editor_grp, priority=12,
                                                  en_save=False)
    result_info = Unicode("", read_only=True).tag(name="Result", group=editor_grp, priority=13, en_save=False)
    layer_thickness = Quantity(Q_(100.0, "µm"),
                               help="Thickness of a known layer. Filled in from the result file when one is "
                                    "chosen").tag(name="Thickness", group=editor_grp, priority=14, en_save=False)
    constant_n = ValueRange([1.5, 0.0], help="n and k of a constant index layer").tag(
        name="Constant n, k", group=editor_grp, priority=15, en_save=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result_cache = {}
        self.set_trait("layer_list", StrList())
        self.layer_list.observe(self._on_layer_selected, "selected_item")
        self._refresh()

    @property
    def _ui_control_widget(self):
        return getattr(self, "_ui_widget_internal", None)

    @_ui_control_widget.setter
    def _ui_control_widget(self, value):
        self._ui_widget_internal = value
        if value:
            self._show_editor_fields()

    @observe("layer_type")
    def _show_editor_fields(self, change=None):
        fields = {LayerType.unknown: [],
                  LayerType.result: ["result_path", "result_info", "layer_thickness"],
                  LayerType.constant: ["layer_thickness", "constant_n"]}
        self.toggle_traits(["layer_type", "layer_label"] + fields[self.layer_type], group_filter=self.editor_grp)

    @observe("layers", "wave_direction")
    def _on_stack_change(self, change):
        self._refresh()

    def _refresh(self, selected_idx=None):
        arrow = "▼" if self.wave_direction == WaveDirection.top_to_bottom else "▲"
        rows = [f"{idx + 1} {arrow} {self._describe(layer)}" for idx, layer in enumerate(self.layers)]
        incident, exit_ = f"Incident medium (air)  {arrow} THz wave", "Exit medium (air)"
        if self.wave_direction == WaveDirection.top_to_bottom:
            rows = [incident] + rows + [exit_]
        else:
            rows = [exit_] + rows + [incident]

        self.layer_list.items = rows
        if selected_idx is not None and 0 <= selected_idx < len(self.layers):
            self.layer_list.selected_item = rows[selected_idx + 1]

        order = [str(idx + 1) for idx in range(len(self.layers))]
        if self.wave_direction == WaveDirection.bottom_to_top:
            order = order[::-1]
        known_d = sum(layer["d"] for layer in self.layers if layer["type"] != LayerType.unknown.name)
        self.set_trait("stack_summary", " → ".join(["incident"] + order + ["exit"])
                       + f"   (known layers {known_d:.1f} µm)")

    @staticmethod
    def _describe(layer):
        label = f"{layer['label']}: " if layer.get("label") else ""
        if layer["type"] == LayerType.unknown.name:
            return f"{label}Unknown (fitted n, k and thickness)"
        if layer["type"] == LayerType.constant.name:
            return f"{label}Known, n = {layer['n']:g} + {layer['k']:g}i, {layer['d']:g} µm"
        return f"{label}Known, {Path(layer['path']).name}, {layer['d']:g} µm"

    def _selected_idx(self):
        match = re.match(r"(\d+) ", self.layer_list.selected_item or "")
        return int(match.group(1)) - 1 if match else None

    def _on_layer_selected(self, change):
        idx = self._selected_idx()
        if idx is None or idx >= len(self.layers):
            return
        layer = self.layers[idx]
        self.layer_type = LayerType[layer["type"]]
        self.layer_label = layer.get("label", "")
        if layer["type"] == LayerType.result.name and Path(layer["path"]).is_file():
            self.result_path = Path(layer["path"])
        if layer["type"] != LayerType.unknown.name:
            self.layer_thickness = Q_(layer["d"], "µm")
        if layer["type"] == LayerType.constant.name:
            self.constant_n = [layer["n"], layer["k"]]

    @observe("result_path")
    def _on_result_path(self, change):
        path = change["new"]
        if not Path(path).is_file():
            self.set_trait("result_info", "")
            return
        try:
            freq, n, d, meas = self._load_result(path)
        except Exception as e:
            self.set_trait("result_info", f"Could not read: {e}")
            return

        if np.isfinite(d):
            self.layer_thickness = Q_(round(float(d), 3), "µm")
        self.set_trait("result_info", f"n = {np.mean(n.real):.3f} + {np.mean(n.imag):.3f}i (mean), "
                                      f"{freq[0]:.2f}–{freq[-1]:.2f} THz, d = {d:.1f} µm ({meas})")

    def _load_result(self, path):
        path = Path(path)
        key = (str(path.resolve()), path.stat().st_mtime)
        if key not in self._result_cache:
            self._result_cache[key] = load_layer_result(path)
        return self._result_cache[key]

    def _layer_from_editor(self):
        layer = {"type": self.layer_type.name, "label": self.layer_label}
        if self.layer_type == LayerType.unknown:
            return layer

        d = float(self.layer_thickness.to("µm").magnitude)
        if d <= 0:
            raise ValueError("The thickness of a known layer must be positive")
        layer["d"] = d
        if self.layer_type == LayerType.constant:
            layer["n"], layer["k"] = float(self.constant_n[0]), float(self.constant_n[1])
        else:
            if not Path(self.result_path).is_file():
                raise ValueError("Choose the result file of the known layer")
            self._load_result(self.result_path)
            layer["path"] = str(Path(self.result_path).resolve())
        return layer

    def _check_unknown_cnt(self, layers):
        if sum(layer["type"] == LayerType.unknown.name for layer in layers) > 1:
            raise ValueError("Only one unknown layer is possible")

    def _apply(self, layers, selected_idx):
        self._check_unknown_cnt(layers)
        self.layers = layers
        self._refresh(selected_idx)

    def _edit(self, func):
        try:
            func()
        except ValueError as e:
            self.logger.warning(str(e))

    @action("Add layer", group=editor_grp, priority=20)
    def add_layer(self):
        def add():
            idx = self._selected_idx()
            insert_idx = len(self.layers) if idx is None else idx + 1
            layers = list(self.layers)
            layers.insert(insert_idx, self._layer_from_editor())
            self._apply(layers, insert_idx)
        self._edit(add)

    @action("Update selected layer", group=editor_grp, priority=21)
    def update_layer(self):
        def update():
            idx = self._selected_idx()
            if idx is None:
                raise ValueError("Select a layer in the list first")
            layers = list(self.layers)
            layers[idx] = self._layer_from_editor()
            self._apply(layers, idx)
        self._edit(update)

    @action("Remove selected layer", group=stack_grp, priority=3)
    def remove_layer(self):
        idx = self._selected_idx()
        if idx is None:
            return
        layers = list(self.layers)
        layers.pop(idx)
        self._apply(layers, min(idx, len(layers) - 1))

    @action("Move layer up", group=stack_grp, priority=4)
    def move_up(self):
        self._move(-1)

    @action("Move layer down", group=stack_grp, priority=5)
    def move_down(self):
        self._move(1)

    def _move(self, step):
        idx = self._selected_idx()
        if idx is None or not 0 <= idx + step < len(self.layers):
            return
        layers = list(self.layers)
        layers[idx], layers[idx + step] = layers[idx + step], layers[idx]
        self._apply(layers, idx + step)

    @action("Clear layers", group=stack_grp, priority=6)
    def clear_layers(self):
        self._apply([], None)

    def propagation_order(self):
        layers = list(self.layers)
        return layers if self.wave_direction == WaveDirection.top_to_bottom else layers[::-1]

    def model_layers(self):
        """Layers in propagation order as used by the multi-layer transmission model."""
        layers = self.propagation_order()
        unknown_cnt = sum(layer["type"] == LayerType.unknown.name for layer in layers)
        if unknown_cnt != 1:
            raise ValueError(f"The layer stack needs exactly one unknown layer, it has {unknown_cnt}")

        model_layers = []
        for layer in layers:
            if layer["type"] == LayerType.unknown.name:
                model_layers.append({"unknown": True})
            elif layer["type"] == LayerType.constant.name:
                model_layers.append({"n": complex(layer["n"], layer["k"]), "d": layer["d"]})
            else:
                if not Path(layer["path"]).is_file():
                    raise ValueError(f"Result file of layer '{self._describe(layer)}' not found")
                freq, n, _, _ = self._load_result(layer["path"])
                model_layers.append({"n": n, "f": freq, "d": layer["d"]})
        return model_layers

    def check_frequency_range(self, freq_axis):
        for layer in self.propagation_order():
            if layer["type"] != LayerType.result.name:
                continue
            freq, _, _, _ = self._load_result(layer["path"])
            if freq_axis[0] < freq[0] - 1e-9 or freq_axis[-1] > freq[-1] + 1e-9:
                logging.warning(f"The fit range {freq_axis[0]:.2f}–{freq_axis[-1]:.2f} THz exceeds the range "
                                f"{freq[0]:.2f}–{freq[-1]:.2f} THz of {Path(layer['path']).name}, "
                                f"its n and k are held constant outside")

    def known_delay(self, freq_axis):
        """Extra delay (ps) of the known layers compared to the same thickness of air, as seen by a linear fit
        of the transmission phase over freq_axis (same estimate as used for the measured group delay)."""
        if len(freq_axis) < 2:
            return 0.0
        w_ = 2 * np.pi * freq_axis
        phase = np.zeros_like(freq_axis)
        for layer in self.model_layers():
            if layer.get("unknown", False):
                continue
            n = np.real(layer_n(layer, freq_axis)) * np.ones_like(freq_axis)
            phase += (n - 1) * layer["d"] * w_ / c_thz
        return abs(np.polyfit(w_, phase, 1)[0])

    def unknown_neighbor_n(self, freq_axis, n1=1, n4=1):
        """Refractive indices before and after the unknown layer in propagation order."""
        layers = self.model_layers()
        idx = next(i for i, layer in enumerate(layers) if layer.get("unknown", False))
        n_prev = layer_n(layers[idx - 1], freq_axis) if idx > 0 else n1
        n_next = layer_n(layers[idx + 1], freq_axis) if idx < len(layers) - 1 else n4
        return n_prev, n_next

    def describe_stack(self):
        return " | ".join(self._describe(layer) for layer in self.propagation_order())

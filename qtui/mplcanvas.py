# -*- coding: utf-8 -*-
"""
This file is part of Taipan.

Copyright (C) 2015 - 2016 Arno Rehn <arno@arnorehn.de>

Taipan is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Taipan is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Taipan.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
from common.units import ureg
from PySide6 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib

def style_mpl():
    _defPal = QtGui.QPalette()
    _defFont = QtGui.QFont()
    if "nt" in os.name:
        _defFont.setFamily("Arial")

    highlightColor = _defPal.color(QtGui.QPalette.Highlight).darker(120)
    darkerHighlightColor = highlightColor.darker(120)
    cycler = matplotlib.cycler('color', [darkerHighlightColor.name(),
                                         highlightColor.name()])

    matplotlib.rc("patch", linewidth=0.5, antialiased=True)
    matplotlib.rc("font", size=10, family=_defFont.family())
    matplotlib.rc("legend", fontsize=10, fancybox=True)
    matplotlib.rc("axes", grid=True, linewidth=1, titlesize='large',
                  axisbelow=True,
                  edgecolor=_defPal.color(QtGui.QPalette.Mid).name(),
                  prop_cycle=cycler)

    matplotlib.rc("grid", linestyle='-',
                  color=_defPal.color(QtGui.QPalette.AlternateBase).name())


class CheckableComboBox(QtWidgets.QComboBox):
    checkedItemsChanged = QtCore.Signal(list)

    def __init__(self, parent=None, label=""):
        super().__init__(parent)
        self._block_signals = False
        self._label = label

        self.setEditable(True)
        self.line_edit = self.lineEdit()
        self.line_edit.setReadOnly(True)
        self.line_edit.installEventFilter(self)

        self.setView(QtWidgets.QListView(self))
        self.setModel(QtGui.QStandardItemModel(self))
        self.view().pressed.connect(self._handle_item_pressed)
        self.model().dataChanged.connect(self._handle_data_changed)

    def _handle_item_pressed(self, index):
        item = self.model().itemFromIndex(index)
        self._block_signals = True
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.CheckState.Checked)
        self._block_signals = False
        self._emit_checked()

    def _handle_data_changed(self, top_left, bottom_right, roles):
        if self._block_signals:
            return

        if QtCore.Qt.ItemDataRole.CheckStateRole in roles:
            self._emit_checked()

    def _emit_checked(self):
        checked = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                checked.append(item.data(QtCore.Qt.ItemDataRole.UserRole))

        self.checkedItemsChanged.emit(checked)

    def addItem(self, text, userData=None):
        item = QtGui.QStandardItem(text)
        item.setData(userData if userData is not None else text, QtCore.Qt.ItemDataRole.UserRole)
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        if self.model().rowCount() == 0:
            item.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.model().appendRow(item)

    def clear(self):
        self.model().clear()

    def hidePopup(self):
        if not self.view().underMouse():
            super().hidePopup()

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        self.setCurrentText(self._label)

        painter.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_ComboBox, opt)
        painter.drawControl(QtWidgets.QStyle.ControlElement.CE_ComboBoxLabel, opt)

    def eventFilter(self, watched, event):
        if watched == self.line_edit and event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.view().isVisible():
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
        return super().eventFilter(watched, event)

class MPLCanvas(QtWidgets.QGroupBox):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    dataset_dict = None

    def __init__(self, parent=None):
        style_mpl()

        super().__init__(parent)

        dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch()
        self.fig = Figure(dpi=dpi)
        self.fig.patch.set_alpha(0)

        self.axes = self.fig.add_subplot(1, 1, 1)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)

        self.mpl_toolbar.addSeparator()

        self.autoscaleAction = self.mpl_toolbar.addAction("Auto-scale")
        self.autoscaleAction.setCheckable(True)
        self.autoscaleAction.setChecked(True)
        self.autoscaleAction.triggered.connect(self._autoscale)

        self.quantity_combobox = CheckableComboBox(self.mpl_toolbar, label="Shown quantities")
        self.quantity_combobox.setMinimumWidth(130)
        self.quantity_combobox.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                             QtWidgets.QSizePolicy.Preferred)
        self.mpl_toolbar.addWidget(self.quantity_combobox)

        self.quantity_combobox.checkedItemsChanged.connect(self.on_checklist_change)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.canvas)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setStretch(0, 1)
        vbox.setStretch(1, 1)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()

        self.fig.tight_layout()

        self.axes.clear()
        self._plotted_artists = {}
        self._line_cmap = {}
        self._checked_order = []

        self._redrawTimer = QtCore.QTimer(self)
        self._redrawTimer.setSingleShot(True)
        self._redrawTimer.setInterval(100)
        self._redrawTimer.timeout.connect(self._redraw)

    def on_checklist_change(self, checked_keys):
        checked_set = set(checked_keys)
        self._checked_order = [k for k in self._checked_order if k in checked_set]
        for k in checked_keys:
            if k not in self._checked_order:
                self._checked_order.append(k)

        for key in self.dataset_dict:
            if key in self._plotted_artists:
                self._plotted_artists[key][0].set_visible(key in checked_set)
                self._plotted_artists[key][1].set_visible(key in checked_set)
            elif key in checked_set:
                dataset = self.dataset_dict[key]
                x_vals = dataset.axes[0].magnitude
                y_vals = dataset.data.magnitude.real
                uncert_vals = dataset.uncert.magnitude.real

                color = self._line_cmap.get(key, "black")

                (line,) = self.axes.plot(x_vals, y_vals, label=key, color=color)

                y_lower = y_vals - uncert_vals
                y_upper = y_vals + uncert_vals

                fill = self.axes.fill_between(
                    x_vals,
                    y_lower,
                    y_upper,
                    color=color,
                    alpha=0.25,
                    linewidth=0
                )

                self._plotted_artists[key] = (line, fill)

        if self.autoscaleAction.isChecked():
            self.axes.relim(visible_only=True)
            self.axes.autoscale()

        self._update_axes_labels()
        self._update_legend()
        self.axes.figure.canvas.draw_idle()

        self.fig.tight_layout()

    def _update_legend(self):
        handles, labels = self.axes.get_legend_handles_labels()
        visible_pairs = [(h, l) for h, l in zip(handles, labels) if h.get_visible()]

        if visible_pairs:
            visible_handles, visible_labels = zip(*visible_pairs)
            self.axes.legend(visible_handles, visible_labels, loc="upper left")
        else:
            legend = self.axes.get_legend()
            if legend is not None:
                legend.remove()

    def _update_axes_labels(self):
        if self._checked_order:
            last_key = self._checked_order[-1]
            dataset = self.dataset_dict[last_key]
            x_unit, y_unit = dataset.axes[0].units, dataset.data.units
            y_unit_str = f"[{y_unit:C~}]" if y_unit != ureg.dimensionless else ""
            x_unit_str = f"[{x_unit:C~}]" if x_unit != ureg.dimensionless else ""
            x_label = f"{dataset.axes_labels[0] if dataset.axes_labels else 'X'} {x_unit_str}"
            y_label = f"{dataset.data_label if dataset.data_label else 'Y'} {y_unit_str}"
        else:
            x_label, y_label = "", ""

        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

    def _redraw(self):
        self.fig.tight_layout()
        self.canvas.draw()
        self.backgrounds = [self.fig.canvas.copy_from_bbox(self.axes.bbox)]

    def showEvent(self, e):
        super().showEvent(e)
        self._redrawTimer.start()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._redrawTimer.start()

    def _autoscale(self, *, redraw=True):
        prev_xlim = self.axes.get_xlim()
        prev_ylim = self.axes.get_ylim()

        self.axes.relim(visible_only=True)
        self.axes.autoscale()

        need_redraw = (prev_xlim != self.axes.get_xlim() or
                       prev_ylim != self.axes.get_ylim())

        if need_redraw and redraw:
            self._redraw()

        return need_redraw

    def _update_combobox(self):
        color_palette = list(mcolors.TABLEAU_COLORS.values())
        self._line_cmap = {
            key: color_palette[i % len(color_palette)]
            for i, key in enumerate(self.dataset_dict.keys())
        }

        previously_checked = []
        if hasattr(self.quantity_combobox, "model"):
            for i in range(self.quantity_combobox.model().rowCount()):
                item = self.quantity_combobox.model().item(i)
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    previously_checked.append(item.data(QtCore.Qt.ItemDataRole.UserRole))

        self.quantity_combobox.clear()
        for k in self.dataset_dict:
            self.quantity_combobox.addItem(k, k)

        if previously_checked:
            for i in range(self.quantity_combobox.model().rowCount()):
                item = self.quantity_combobox.model().item(i)
                key_name = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if key_name in previously_checked:
                    item.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            if self.quantity_combobox.model().rowCount() > 0:
                first_item = self.quantity_combobox.model().item(0)
                first_item.setCheckState(QtCore.Qt.CheckState.Checked)

    def set_dataset_dict(self, new_dataset_dict):
        previous_dict = self.dataset_dict
        self.dataset_dict = new_dataset_dict

        self.axes.clear()
        self._plotted_artists = {}

        if previous_dict is None or (list(map(str, previous_dict)) != list(map(str, new_dataset_dict))):
            self._update_combobox()

        self.quantity_combobox._emit_checked()

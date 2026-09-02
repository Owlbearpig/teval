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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setView(QtWidgets.QListView(self))
        self.setModel(QtGui.QStandardItemModel(self))

        self.view().pressed.connect(self._handle_item_pressed)
        self._changed_lock = False

    def _handle_item_pressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.CheckState.Checked)

        self._emit_checked()

    def _emit_checked(self):
        checked = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                checked.append(item.data(QtCore.Qt.ItemDataRole.UserRole))

        self.checkedItemsChanged.emit(checked)

    def addItem(self, text, data=None):
        item = QtGui.QStandardItem(text)
        item.setData(data if data is not None else text, QtCore.Qt.ItemDataRole.UserRole)
        item.setCheckable(False)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.model().appendRow(item)
        self._emit_checked()

    def clear(self):
        self.model().clear()
        self._emit_checked()

    def hidePopup(self):
        if not self.view().underMouse():
            super().hidePopup()

class MPLCanvas(QtWidgets.QGroupBox):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    quantity_dict = None
    activeDataSets = {}

    _prevAxesLabels = None
    _axesLabel = None
    _prevDataLabel = None
    _dataLabel = None

    _lastPlotTime = 0
    _isLiveData = False

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

        self.mpl_toolbar.addWidget(QtWidgets.QLabel("Selected quantities: "))

        self.quantity_combobox = CheckableComboBox(self.mpl_toolbar)
        self.quantity_combobox.setEditable(True)
        self.quantity_combobox.lineEdit().setReadOnly(True)
        self.quantity_combobox.setMinimumWidth(40)
        self.quantity_combobox.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                             QtWidgets.QSizePolicy.Preferred)
        self.mpl_toolbar.addWidget(self.quantity_combobox)

        self.quantity_combobox.checkedItemsChanged.connect(self._select_datasets)

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
        self._plot_lines = {}
        self._line_cmap = {}

        self._redrawTimer = QtCore.QTimer(self)
        self._redrawTimer.setSingleShot(True)
        self._redrawTimer.setInterval(100)
        self._redrawTimer.timeout.connect(self._redraw)

    def update_combobox(self, quantity_dict=None):
        if quantity_dict is None:
            quantity_dict = self.quantity_dict
        self.quantity_combobox.clear()
        for k in quantity_dict:
            self.quantity_combobox.addItem(k, k)

    def _select_datasets(self, checked_keys):
        quantity_key = self.quantity_combobox.currentData()
        if quantity_key is None:
            return
        self.activeDataSets = {k: self.quantity_dict[k] for k in checked_keys if k in self.quantity_dict}

        self._replot(redraw_axes_labels=True)


    def _redraw(self):
        self.fig.tight_layout()
        self.canvas.draw()
        self.backgrounds = [self.fig.canvas.copy_from_bbox(self.axes.bbox)]

    def _redraw(self):
        self.fig.tight_layout()
        self.canvas.draw()

    def showEvent(self, e):
        super().showEvent(e)
        self._redrawTimer.start()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._redrawTimer.start()

    def _dataSetToLines(self, data, line):
        if data is None or not data.axes:
            line.set_data([], [])
            return

        # data.data -= np.mean(data.data)
        line.set_data(data.axes[0].magnitude, data.data.magnitude)

    def _autoscale(self, *, redraw=True):
        prev_xlim = self.axes.get_xlim()
        prev_ylim = self.axes.get_ylim()

        self.axes.relim()
        self.axes.autoscale()

        need_redraw = (prev_xlim != self.axes.get_xlim() or
                       prev_ylim != self.axes.get_ylim())

        if need_redraw and redraw:
            self._redraw()

        return need_redraw

    def _replot(self, redraw_axes_labels=True):
        current_keys = set(self.activeDataSets.keys())
        for existing_key in list(self._plot_lines.keys()):
            if existing_key not in current_keys:
                self._plot_lines[existing_key].remove()
                del self._plot_lines[existing_key]

        for key, dataset in self.activeDataSets.items():
            if dataset is not None and dataset.axes:
                x_vals = dataset.axes[0].magnitude
                y_vals = dataset.data.magnitude.real

                if key in self._plot_lines:
                    self._plot_lines[key].set_data(x_vals, y_vals)
                else:
                    color = self._line_cmap.get(key, "black")
                    line, = self.axes.plot(x_vals, y_vals, label=key, color=color)
                    self._plot_lines[key] = line

        if self.activeDataSets:
            last_ds_name = list(self.activeDataSets.keys())[-1]
            last_ds = self.activeDataSets[last_ds_name]
            if redraw_axes_labels and last_ds.axes:
                x_label = f"{last_ds.axes_labels[0] if last_ds.axes_labels else 'X'} [{last_ds.axes[0].units:C~}]"
                y_label = f"{last_ds.data_label if last_ds.data_label else last_ds_name} [{last_ds.data.units:C~}]"

                self.axes.set_xlabel(x_label)
                self.axes.set_ylabel(y_label)
            if self._plot_lines:
                self.axes.legend(loc="upper right")
        else:
            if self._plot_lines:
                self.axes.legend().remove()

        if self.autoscaleAction.isChecked():
            self.axes.relim()
            self.axes.autoscale()

        self.fig.tight_layout()
        self.canvas.draw()

    def set_canvas_values(self, quantity_dict, axes_labels, data_label):
        self.quantity_dict = quantity_dict

        color_palette = list(mcolors.TABLEAU_COLORS.values())
        self._line_cmap = {
            key: color_palette[i % len(color_palette)]
            for i, key in enumerate(self.quantity_dict.keys())
        }

        previously_checked = []
        if hasattr(self.quantity_combobox, "model"):
            for i in range(self.quantity_combobox.model().rowCount()):
                item = self.quantity_combobox.model().item(i)
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    previously_checked.append(item.data(QtCore.Qt.ItemDataRole.UserRole))

        self.quantity_combobox.clear()
        for k in self.quantity_dict:
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

        self.quantity_combobox._emit_checked()

        self._axesLabel = axes_labels
        self._dataLabel = data_label

        self._replot(redraw_axes_labels=True)


class MPLCanvasDoubleDataset(MPLCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



from PySide6 import QtWidgets, QtCore, QtGui
from traitlets import Integer, Float, Unicode, Bool, Enum
from common.measurement_selection import MeasurementSelection
from common.settings import Settings
from common.eval_component.eval_result import EvalResult
from qtui.changeindicatorspinbox import ChangeIndicatorSpinBox
from qtui.changeindicatorlineedit import ChangeIndicatorLineEdit
from qtui.fastfilefilterproxy import FastNameFilterProxyModel
from common.components import ComponentBase
from traitlets import Instance
from qtui.mplcanvas import FigureCanvasWidget
from common.traits import (Quantity, Path as PathTrait, ValueRange, MultiPathSelection, StrListSelection, PlotFigure,
                           FitParameterTable)
from dataclasses import replace
from pathlib import Path
import types
import logging
from collections import OrderedDict
from itertools import chain
import numpy as np
from qtui.flowlayout import FlowLayout
from common.traits import Q_, QuantityDict
from qtui.mplcanvas import MPLCanvas

plot_traits = (QuantityDict, PlotFigure)

def is_component_trait(x):
    return isinstance(x, Instance) and isinstance(x.klass, type) and issubclass(x.klass, ComponentBase)


def create_number_entry(component, name, trait):
    raw_value = trait.get(component)
    is_range = isinstance(raw_value, (tuple, list))
    values = list(raw_value) if is_range else [raw_value]
    sb_cnt = len(values)
    sample = values[0]

    is_integer = isinstance(trait, Integer) or isinstance(sample, int)
    is_quantity = isinstance(trait, Quantity) or isinstance(sample, Q_)

    is_double_spinbox = not is_integer

    def limit(bound, default):
        if bound is None:
            return default
        return bound.magnitude if is_quantity else bound

    if is_integer:
        min_val = limit(trait.min, -2147483648)
        max_val = limit(trait.max, 2147483647)
    else:
        min_val = limit(trait.min, float('-inf'))
        max_val = limit(trait.max, float('inf'))

    significant_figures = trait.metadata.get('significant_figures', 3)

    if is_quantity:
        preferred = trait.metadata.get('preferred_units', None)
        units = [preferred or v.units for v in values]
    else:
        units = [None] * sb_cnt

    def get_value(idx):
        val = trait.get(component)
        val = val[idx] if is_range else val
        return val.to(units[idx]).magnitude if is_quantity else val

    def set_value(idx, number):
        new_val = number * units[idx] if is_quantity else number
        if is_range:
            full = list(trait.get(component))
            full[idx] = new_val
            setattr(component, name, full)
        else:
            setattr(component, name, new_val)

    layout = QtWidgets.QHBoxLayout()

    spinboxes = []
    def setup_single_spinbox(sb_idx):
        spinbox = ChangeIndicatorSpinBox(
            is_double_spinbox=is_double_spinbox,
            actual_value_getter=lambda: get_value(sb_idx))
        spinboxes.append(spinbox)

        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setToolTip(trait.help)
        spinbox.setReadOnly(trait.read_only)
        if trait.read_only:
            spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

        if is_quantity:
            spinbox.setSuffix(f" {units[sb_idx]:C~}")

        if is_double_spinbox:
            def textFromValue(self, val):
                if abs(val) < 10 ** (1 - significant_figures) and val != 0.0:
                    text = f"{val:.{significant_figures}e}"
                else:
                    text = f"{val:.{significant_figures}f}"
                return text.replace(".", ",")

            def valueFromText(self, text):
                clean_text = text.replace(self.suffix(), '').strip()
                try:
                    return float(clean_text.replace(",", "."))
                except ValueError:
                    return 0.0

            spinbox.textFromValue = types.MethodType(textFromValue, spinbox)
            spinbox.valueFromText = types.MethodType(valueFromText, spinbox)

            def sizeHint(self):
                original_hint = QtWidgets.QDoubleSpinBox.sizeHint(self)
                font_metrics = self.fontMetrics()
                text_width = font_metrics.horizontalAdvance(
                    f"{get_value(sb_idx):.{significant_figures}f}")
                button_padding = 30
                suffix_padding = font_metrics.horizontalAdvance(self.suffix())
                new_width = text_width + button_padding + suffix_padding

                return QtCore.QSize(new_width, original_hint.height())

            spinbox.sizeHint = types.MethodType(sizeHint, spinbox)
            spinbox.updateGeometry()

        layout.addWidget(spinbox)

    for i in range(sb_cnt):
        setup_single_spinbox(sb_idx=i)

        if i != sb_cnt - 1:
            separator_label = QtWidgets.QLabel("-")
            separator_label.setAlignment(QtCore.Qt.AlignCenter)
            separator_label.setStyleSheet("padding: 0 1px;")
            layout.addWidget(separator_label)

    if not trait.read_only:
        apply = QtWidgets.QToolButton()
        apply.setFocusPolicy(QtCore.Qt.NoFocus)
        apply.setText('✓')
        apply.setAutoRaise(True)
        layout.addWidget(apply)

        def apply_all():
            if is_range:
                full = [sb.value() * units[i] if is_quantity else sb.value()
                        for i, sb in enumerate(spinboxes)]
                setattr(component, name, full)
            else:
                set_value(0, spinboxes[0].value())
            refresh_spinboxes()

        apply.clicked.connect(apply_all)

        def connect_spinbox(sb_idx, spinbox):
            def apply_single():
                set_value(sb_idx, spinbox.value())

            spinbox.editingFinished.connect(apply_single)
            spinbox.editingFinished.connect(spinbox.check_changed)

        for idx, sb in enumerate(spinboxes):
            connect_spinbox(idx, sb)

    def refresh_spinboxes(*args):
        for sb_idx, spinbox in enumerate(spinboxes):
            spinbox.blockSignals(True)
            spinbox.setValue(get_value(sb_idx))
            spinbox.blockSignals(False)
            spinbox.check_changed()

    refresh_spinboxes()
    component.observe(refresh_spinboxes, name)

    layout.setContentsMargins(0, 0, 0, 0)
    layout.setStretch(0, 1)
    layout.setStretch(1, 0)

    return layout


def create_progressbar(component, name, trait):
    progressBar = QtWidgets.QProgressBar()
    progressBar.setMinimum(trait.min * 1000)
    progressBar.setMaximum(trait.max * 1000)
    progressBar.setValue(int(trait.get(component) * 1000))
    component.observe(
        lambda change: progressBar.setValue(int(change['new'] * 1000)),
        name
    )

    return progressBar


def create_checkbox(component, name, trait):
    checkbox = QtWidgets.QCheckBox()
    checkbox.setChecked(trait.get(component))
    checkbox.setEnabled(not trait.read_only)
    checkbox.setToolTip(trait.help)
    component.observe(lambda change: checkbox.setChecked(change['new']), name)
    if not trait.read_only:
        checkbox.toggled.connect(lambda toggled:
                                 setattr(component, name, toggled))

    return checkbox


def create_action(component, action):
    qaction = QtGui.QAction(action.metadata.get('name', action.__name__), None)
    qaction.setToolTip(action.help)

    qaction.triggered.connect(lambda: action())

    return qaction


def create_combobox(component, name, trait):
    combobox = QtWidgets.QComboBox()

    def populate_items():
        combobox.blockSignals(True)
        combobox.clear()

        current_trait = component.traits()[name]
        for item in current_trait.values:
            if hasattr(item, "value") and isinstance(item.value, str):
                item_name = str(item.value)
            elif hasattr(item, "name"):
                item_name = item.name
            else:
                item_name = str(item)
            combobox.addItem(item_name, item)

        current_val = current_trait.get(component)
        if hasattr(current_val, "value") and isinstance(current_val.value, str):
            text = str(current_val.value)
        elif hasattr(current_val, "name"):
            text = current_val.name
        else:
            text = str(current_val)

        combobox.setCurrentText(text)

        combobox.blockSignals(False)

    populate_items()

    def update_combobox(change):
        populate_items()

    component.observe(update_combobox, name)

    combobox.currentIndexChanged.connect(
        lambda: setattr(component, name, combobox.currentData())
    )

    return combobox


def create_label(component, name, trait):
    label = QtWidgets.QLabel()
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setText(trait.get(component))
    label.setToolTip(trait.help)

    def on_change(change):
        QtCore.QMetaObject.invokeMethod(
            label,
            "setText",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, str(change['new']))
        )

    component.observe(on_change, name)

    return label


def create_lineedit(component, name, trait):
    lineEdit = ChangeIndicatorLineEdit(actual_value_getter=lambda: trait.get(component))
    lineEdit.setText(trait.get(component))
    lineEdit.setToolTip(trait.help)

    def apply_text_to_lineedit(change):
        lineEdit.blockSignals(True)
        lineEdit.setText(change['new'])
        lineEdit.blockSignals(False)

    def apply_text_to_component():
        setattr(component, name, lineEdit.text())

    component.observe(apply_text_to_lineedit, name)
    lineEdit.editingFinished.connect(apply_text_to_component)
    lineEdit.editingFinished.connect(lineEdit.check_changed)

    return lineEdit


class MultilineEdit(QtWidgets.QPlainTextEdit):
    apply_requested = QtCore.Signal()

    def keyPressEvent(self, event):
        if (event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)
                and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            self.apply_requested.emit()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        self.apply_requested.emit()
        super().focusOutEvent(event)


def create_multiline_edit(component, name, trait):
    layout = QtWidgets.QHBoxLayout()
    text_edit = MultilineEdit()
    text_edit.setPlainText(trait.get(component))
    text_edit.setToolTip(trait.help)
    text_edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    text_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
    line_height = text_edit.fontMetrics().lineSpacing()
    text_edit.setMinimumHeight(5 * line_height)
    text_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)

    def update_changed_indicator():
        changed = text_edit.toPlainText() != trait.get(component)
        text_edit.setStyleSheet("QPlainTextEdit { border: 1px solid #E6A100; }" if changed else "")

    def apply_text_to_component():
        if text_edit.toPlainText() != trait.get(component):
            setattr(component, name, text_edit.toPlainText())
        update_changed_indicator()

    def apply_text_to_edit(change):
        if change["new"] != text_edit.toPlainText():
            text_edit.blockSignals(True)
            text_edit.setPlainText(change["new"])
            text_edit.blockSignals(False)
        update_changed_indicator()

    component.observe(apply_text_to_edit, name)
    text_edit.apply_requested.connect(apply_text_to_component)
    text_edit.textChanged.connect(update_changed_indicator)

    apply = QtWidgets.QToolButton()
    apply.setFocusPolicy(QtCore.Qt.NoFocus)
    apply.setText('✓')
    apply.setAutoRaise(True)
    apply.setToolTip("Apply (Ctrl+Enter)")
    apply.clicked.connect(apply_text_to_component)

    layout.addWidget(text_edit)
    layout.addWidget(apply, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setStretch(0, 1)

    return layout


class LatexView(QtWidgets.QScrollArea):
    # renders the latex lines (one formula per line) of a Unicode trait with matplotlib mathtext
    refresh_requested = QtCore.Signal()

    def __init__(self, component, name, trait):
        super().__init__()
        self.component, self.name = component, name
        self.setWidgetResizable(True)
        self.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setWidget(self.label)

        self.refresh_requested.connect(self.refresh)
        component.observe(lambda change: self.refresh_requested.emit(), name)
        self.refresh()

    def _render_line(self, latex, color, dpi):
        from io import BytesIO
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        buffer = BytesIO()
        fig = Figure()
        FigureCanvasAgg(fig)
        fig.text(0, 0, f"${latex}$", color=color, fontsize=max(self.font().pointSizeF(), 9))
        fig.savefig(buffer, dpi=dpi, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
        image = QtGui.QImage.fromData(buffer.getvalue(), "PNG")
        image.setDevicePixelRatio(self.devicePixelRatioF())
        return image

    @QtCore.Slot()
    def refresh(self):
        lines = [line for line in getattr(self.component, self.name).split("\n") if line.strip()]
        self.label.setToolTip("\n".join(lines))
        if not lines:
            self.label.clear()
            self.setFixedHeight(0)
            return

        color = self.palette().color(QtGui.QPalette.ColorRole.WindowText).name()
        dpi = 1.3 * self.logicalDpiY() * self.devicePixelRatioF()
        try:
            images = [self._render_line(line, color, dpi) for line in lines]
        except Exception as e:
            logging.warning(f"Could not render formula: {e}")
            self.label.setText("\n".join(lines))
            self.setFixedHeight(self.label.sizeHint().height() + 4)
            return

        spacing = int(0.4 * self.fontMetrics().height() * self.devicePixelRatioF())
        width = max(image.width() for image in images)
        height = sum(image.height() for image in images) + spacing * (len(images) - 1)
        pixmap = QtGui.QPixmap(width, height)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        y = 0
        for image in images:
            image.setDevicePixelRatio(1.0)
            painter.drawImage(0, y, image)
            y += image.height() + spacing
        painter.end()
        pixmap.setDevicePixelRatio(self.devicePixelRatioF())

        self.label.setPixmap(pixmap)
        scrollbar = self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(int(height / self.devicePixelRatioF()) + scrollbar + 4)


def create_path_selector(component, name, prettyName, trait):
    layout = QtWidgets.QHBoxLayout()

    def get_current_path():
        return str(trait.get(component))

    lineEdit = ChangeIndicatorLineEdit(actual_value_getter=get_current_path)
    lineEdit.setText(str(trait.get(component)))
    lineEdit.setToolTip(trait.help)

    def apply_path_to_lineedit(change):
        lineEdit.blockSignals(True)
        lineEdit.setText(str(change['new']))
        lineEdit.blockSignals(False)

    def apply_path_to_component():
        try:
            setattr(component, name, Path(lineEdit.text()))
        except Exception as e:
            logging.error(e)
            lineEdit.setText(get_current_path())

    component.observe(apply_path_to_lineedit, name)
    lineEdit.editingFinished.connect(apply_path_to_component)
    lineEdit.editingFinished.connect(lineEdit.check_changed)

    choose = QtWidgets.QToolButton()
    choose.setFocusPolicy(QtCore.Qt.NoFocus)
    choose.setText('...')
    choose.setAutoRaise(True)
    choose.setEnabled(not trait.read_only)

    def choose_path():
        if trait.is_dir and not trait.is_file:
            name = QtWidgets.QFileDialog.getExistingDirectory(
                       caption="Choose " + prettyName)

        else:
            if trait.must_exist:
                name, filt = QtWidgets.QFileDialog.getOpenFileName(
                                 caption="Choose " + prettyName)
            else:
                name, filt = QtWidgets.QFileDialog.getSaveFileName(
                                 caption="Choose " + prettyName)

        if not name:
            return

        lineEdit.setText(name)
        apply_path_to_component()
        lineEdit.check_changed()

    choose.clicked.connect(choose_path)

    def sizeHint(self):
        original_hint = QtWidgets.QLineEdit.sizeHint(self)

        user_width = trait.metadata.get("width", None)
        og_width = original_hint.width()
        width = og_width if user_width is None else max(og_width, user_width)

        return QtCore.QSize(width, original_hint.height())

    lineEdit.sizeHint = types.MethodType(sizeHint, lineEdit)
    lineEdit.updateGeometry()

    layout.addWidget(lineEdit)
    layout.addWidget(choose)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setStretch(0, 1)
    layout.setStretch(1, 0)

    return layout


def create_tree_path_selector(component, name, prettyName, trait):
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    model = QtWidgets.QFileSystemModel()
    root_path = getattr(trait.get(component), "root_path", None)
    initial_root_path = str(root_path) if root_path is not None else QtCore.QDir.homePath()
    model.setRootPath(initial_root_path)

    proxy = FastNameFilterProxyModel()
    proxy.setSourceModel(model)

    shown_filenames = getattr(trait.get(component), "shown_filenames", None)
    proxy.set_allowed_filenames(shown_filenames)

    tree = QtWidgets.QTreeView()
    tree.setModel(proxy)
    tree.hideColumn(1)
    tree.hideColumn(2)
    tree.hideColumn(3)
    tree.setRootIndex(proxy.mapFromSource(model.index(initial_root_path)))
    tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    tree.setMinimumHeight(180)

    def apply_initial_selection():
        initially_selected_paths = [str(p) for p in getattr(component, name).selected_paths]
        if not initially_selected_paths:
            return

        selection = QtCore.QItemSelection()
        for path_str in initially_selected_paths:
            src_idx = model.index(path_str)
            if src_idx.isValid():
                proxy_idx = proxy.mapFromSource(src_idx)
                if proxy_idx.isValid():
                    selection.select(proxy_idx, proxy_idx)

        if not selection.isEmpty():
            sel_model = tree.selectionModel()

            sel_model.blockSignals(True)
            sel_model.select(
                selection,
                QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Rows
            )

            sel_model.blockSignals(False)

    model.directoryLoaded.connect(lambda: apply_initial_selection())

    def update_trait_selection():
        selected_proxy_indexes = tree.selectionModel().selectedRows(column=0)
        selected_indexes = [proxy.mapToSource(idx) for idx in selected_proxy_indexes]
        paths = [Path(model.filePath(idx)) for idx in selected_indexes]
        multi_path_class = getattr(component, name)
        multi_path_class.selected_paths = paths

    tree.selectionModel().selectionChanged.connect(lambda *args: update_trait_selection())

    def update_root(change):
        new_val = change["new"]
        new_path_str = str(getattr(new_val, "root_path"))
        new_filenames = getattr(new_val, "shown_filenames", None)
        proxy.set_allowed_filenames(new_filenames)
        model.setRootPath(new_path_str)
        tree.setRootIndex(proxy.mapFromSource(model.index(new_path_str)))

    component.observe(update_root, name)
    layout.addWidget(tree)

    return container

def create_list_view(component, name, trait):
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    model = QtCore.QStringListModel()

    list_view = QtWidgets.QListView()
    list_view.setModel(model)
    if trait.read_only:
        list_view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    if "max_width" in trait.metadata:
        list_view.setMaximumWidth(trait.metadata["max_width"])
    if "min_width" in trait.metadata:
        list_view.setMinimumWidth(trait.metadata["min_width"])

    layout.addWidget(list_view)
    str_list = getattr(component, name)

    def update_list(change):
        string_list = [str(s) for s in change["new"]]
        model.setStringList(string_list)

        if string_list:
            first_index = model.index(0, 0)
            list_view.selectionModel().setCurrentIndex(
                first_index,
                QtCore.QItemSelectionModel.ClearAndSelect
            )
            str_list.selected_item = string_list[0]

    str_list.observe(update_list, "items")

    def update_trait_selection():
        selected_indexes = list_view.selectionModel().selectedRows(column=0)
        if not selected_indexes:
            return

        str_list.selected_item = selected_indexes[0].data()

    list_view.selectionModel().selectionChanged.connect(lambda *args: update_trait_selection())

    def update_list_selection(change):
        new_item = change["new"]
        if new_item is None:
            list_view.selectionModel().clearSelection()
            return

        items = str_list.items
        if new_item in items:
            row = items.index(new_item)
            selected_item_index = model.index(row, 0)

            list_view.selectionModel().blockSignals(True)
            list_view.selectionModel().setCurrentIndex(
                selected_item_index,
                QtCore.QItemSelectionModel.ClearAndSelect
            )
            list_view.selectionModel().blockSignals(False)

    str_list.observe(update_list_selection, "selected_item")

    return container

class ParameterTableWidget(QtWidgets.QTableWidget):
    refresh_requested = QtCore.Signal()

    def __init__(self, component, name, trait):
        super().__init__()
        self.component, self.name, self.read_only = component, name, trait.read_only
        if self.read_only:
            self.columns = ["Parameter", "Value", "Unit"]
        else:
            self.columns = ["Parameter", "Unit", "Lower bound", "Upper bound"]

        self.setColumnCount(len(self.columns))
        self.setHorizontalHeaderLabels(self.columns)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.setToolTip(trait.help)
        self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)

        self.refresh_requested.connect(self.refresh)
        component.observe(lambda change: self.refresh_requested.emit(), name)
        self.itemChanged.connect(self.on_item_changed)
        self.refresh()

    @staticmethod
    def _fmt(val):
        return "" if val is None else f"{val:.6g}"

    def _row_texts(self, p):
        if self.read_only:
            return [p.display_name, self._fmt(p.value), p.unit]
        return [p.display_name, p.unit, self._fmt(p.lower), self._fmt(p.upper)]

    @QtCore.Slot()
    def refresh(self):
        parameters = getattr(self.component, self.name)
        self.blockSignals(True)
        self.setRowCount(len(parameters))
        for row, p in enumerate(parameters):
            for col, text in enumerate(self._row_texts(p)):
                item = QtWidgets.QTableWidgetItem(text)
                if self.read_only or col == 0:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if col == 0 and p.label:
                    item.setToolTip(p.name)
                self.setItem(row, col, item)
        self.blockSignals(False)

        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(row) for row in range(self.rowCount()))
        self.setFixedHeight(height)

    def on_item_changed(self, item):
        parameters = list(getattr(self.component, self.name))
        row, col = item.row(), item.column()
        p, text = parameters[row], item.text().strip()
        try:
            if col == 1:
                Q_(1, text)  # check that the unit is known
                parameters[row] = replace(p, unit=text)
            else:
                number = float(text.replace(",", "."))
                parameters[row] = replace(p, **{"lower" if col == 2 else "upper": number})
            setattr(self.component, self.name, tuple(parameters))
        except Exception as e:
            logging.error(f"Invalid value '{text}' for {self.columns[col].lower()} of {p.display_name}: {e}")
            self.refresh()


def create_parameter_table(component, name, trait):
    return ParameterTableWidget(component, name, trait)


def create_plot_area(component, name, prettyName, trait):
    def draw(change):
        canvas.set_dataset_dict(change["new"])

    canvas = MPLCanvas()

    component.observe(draw, name)

    canvas.setTitle(prettyName)

    initial_value = trait.get(component)
    if initial_value:
        canvas.set_dataset_dict(initial_value)

    return canvas

def create_figure_area(component, name, prettyName, trait):
    w = FigureCanvasWidget(trait.get(component))
    w.setTitle(prettyName)
    return w

def _group(trait):
    return trait.metadata.get("group", "General")


def _prettyName(trait, name):
    return trait.metadata.get("name", name)


traitPriority = {
    'Unicode': -1,
    'Path': 0,
    'Float': 1,
    'Int': 1,
    'Quantity': 1,
    'ValueRange': 1,
    'Enum': 2,
    'Bool': 7,
    'Float_readonly': 10
}


def _traitSortingKey(args):
    name, trait = args
    traittype = type(trait).__name__
    traittype_ro = traittype + "_readonly"

    prio = traitPriority.get(traittype_ro, None)
    if prio is None:
        prio = traitPriority.get(traittype, None)
    if prio is None:
        prio = 999

    userPrio = trait.metadata.get("priority", 999)

    return userPrio, prio, name


def generate_component_ui(name, component):
    controlWidget = QtWidgets.QWidget()

    # filter and sort traits
    traits = [(name, trait) for name, trait
              in sorted(chain(component.traits().items(),
                              component.actions), key=_traitSortingKey)
              if not is_component_trait(trait)]

    groups = OrderedDict()
    hasPlots = False
    for name, trait in traits:
        if isinstance(trait, plot_traits):
            hasPlots = True
            continue

        group = _group(trait)

        if group not in groups:
            box = QtWidgets.QGroupBox(group, controlWidget)
            QtWidgets.QFormLayout(box)
            groups[group] = box

    controlWidget.param_widgets = {}
    for name, trait in traits:
        if isinstance(trait, plot_traits):
            continue

        prettyName = _prettyName(trait, name)
        group = _group(trait)
        layout = groups[group].layout()
        if trait.metadata.get("fullwidth", False):
            groups[group].fullwidth = True

        groups[group].combine = trait.metadata.get("combine", False)

        field_widget = None
        if isinstance(trait, (Quantity, Integer, ValueRange)):
            field_widget = create_number_entry(component, name, trait)
        elif isinstance(trait, Enum) and not trait.read_only:
            field_widget = create_combobox(component, name, trait)
        elif isinstance(trait, Float):
            if trait.read_only and not (np.isinf(trait.min) or np.isinf(trait.max)):
                field_widget = create_progressbar(component, name, trait)
            else:
                field_widget = create_number_entry(component, name, trait)
        elif isinstance(trait, Bool):
            field_widget = create_checkbox(component, name, trait)
        elif isinstance(trait, Unicode) and trait.metadata.get("latex", False):
            field_widget = LatexView(component, name, trait)
        elif isinstance(trait, Unicode) and trait.metadata.get("multiline", False) and not trait.read_only:
            field_widget = create_multiline_edit(component, name, trait)
        elif isinstance(trait, Unicode):
            if trait.read_only:
                field_widget = create_label(component, name, trait)
            else:
                field_widget = create_lineedit(component, name, trait)
        elif isinstance(trait, StrListSelection):
            field_widget = create_list_view(component, name, trait)
        elif isinstance(trait, PathTrait):
                field_widget = create_path_selector(component, name, prettyName, trait)
        elif isinstance(trait, MultiPathSelection):
            field_widget = create_tree_path_selector(component, name, prettyName, trait)
        elif isinstance(trait, FitParameterTable):
            field_widget = create_parameter_table(component, name, trait)

        if field_widget:
            disable_label = trait.metadata.get("disable_label", False)
            if isinstance(trait, (MultiPathSelection, StrListSelection)) or disable_label:
                label_widget = None
                layout.addRow(field_widget)
            else:
                label_widget = QtWidgets.QLabel(prettyName + ": ")
                layout.addRow(label_widget, field_widget)
            controlWidget.param_widgets[name] = (label_widget, field_widget)

        if callable(trait):
            qaction = create_action(component, trait)
            qaction.setParent(controlWidget)
            btn = QtWidgets.QToolButton()
            btn.setDefaultAction(qaction)
            layout.addRow(None, btn)

    controlLayout = FlowLayout(controlWidget)
    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setFrameStyle(QtWidgets.QFrame.NoFrame)
    scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
    scrollArea.setWidgetResizable(True)

    for group in groups.values():
        if not group.combine:
            controlLayout.addWidget(group)
    if isinstance(component, EvalResult):
        vSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vSplitter.setChildrenCollapsible(False)

        hSplitter = QtWidgets.QSplitter()
        hSplitter.setStretchFactor(1, 0)
        hSplitter.setStretchFactor(0, 1)
        hSplitter.setChildrenCollapsible(False)
        for group in groups.values():
            if group.combine:
                hSplitter.addWidget(group)

        vSplitter.addWidget(hSplitter)
        vSplitter.addWidget(controlWidget)

        vSplitter.setStretchFactor(0, 0)
        vSplitter.setStretchFactor(1, 1)
        scrollArea.setWidget(vSplitter)
        vSplitter.setSizes([200, 400])
    elif isinstance(component, MeasurementSelection):
        vSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vSplitter.setChildrenCollapsible(False)

        hSplitter = QtWidgets.QSplitter()
        hSplitter.setStretchFactor(1, 0)
        hSplitter.setStretchFactor(0, 1)
        hSplitter.setChildrenCollapsible(False)
        for group in groups.values():
            if group.combine:
                hSplitter.addWidget(group)

        vSplitter.addWidget(controlWidget)
        vSplitter.addWidget(hSplitter)
        vSplitter.setStretchFactor(0, 0)
        vSplitter.setStretchFactor(1, 1)
        scrollArea.setWidget(vSplitter)
        vSplitter.setSizes([0, 10000])
    else:
        scrollArea.setWidget(controlWidget)


    class ViewportResizeFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Resize:
                margins = controlWidget.layout().contentsMargins()
                viewport_width = obj.width() - margins.left() - margins.right()

                for box in groups.values():
                    if getattr(box, "fullwidth", False):
                        box.setFixedWidth(max(viewport_width, 0))
                    elif getattr(box, "halfwidth", False):
                        box.setFixedWidth(max(viewport_width/2, 0))

            return super().eventFilter(obj, event)

    controlWidget._resize_filter = ViewportResizeFilter()
    scrollArea.viewport().installEventFilter(controlWidget._resize_filter)
    scrollArea.setMinimumWidth(scrollArea.sizeHint().width())

    if not groups:
        scrollArea.hide()

    component._ui_control_widget = controlWidget
    if not hasPlots:
        return scrollArea

    plotWidget = QtWidgets.QWidget()
    plotBox = QtWidgets.QVBoxLayout(plotWidget)
    plotBox.setContentsMargins(0, 0, 0, 0)

    for name, trait in traits:
        if not isinstance(trait, plot_traits):
            continue
        prettyName = _prettyName(trait, name)

        if isinstance(trait, QuantityDict):
            plotBox.addWidget(create_plot_area(component, name, prettyName, trait))
        elif isinstance(trait, PlotFigure):
            plotBox.addWidget(create_figure_area(component, name, prettyName, trait))

    hSplitter = QtWidgets.QSplitter()
    hSplitter.addWidget(plotWidget)
    hSplitter.addWidget(scrollArea)
    scrollArea.setMinimumWidth(0)
    hSplitter.setStretchFactor(0, 1)
    hSplitter.setStretchFactor(1, 1)
    hSplitter.setSizes([1, 100])
    hSplitter.setChildrenCollapsible(False)

    return hSplitter

def generate_ui(component):
    stack = QtWidgets.QStackedWidget()
    toggle_source = component.settings.component_toggle

    def make_tree_items(component, name, depth, treeitem, toggle_trait=None):
        if component is None:
            return
        prettyName = component.object_name or name
        newItem = QtWidgets.QTreeWidgetItem(treeitem)
        newItem.setText(0, prettyName)
        if isinstance(component, Settings):
            newItem.setExpanded(False)
        else:
            newItem.setExpanded(True)

        widget = generate_component_ui(prettyName, component)
        newItem.widgetId = stack.addWidget(widget)

        if toggle_trait is not None:
            toggle_name = toggle_trait.name

            def apply_visibility(change=None, item=newItem, tn=toggle_name):
                visible = bool(getattr(toggle_source, tn))
                item.setHidden(not visible)
                if not visible and stack.currentIndex() == item.widgetId:
                    stack.setCurrentIndex(0)

            apply_visibility()
            toggle_source.observe(apply_visibility, toggle_name)

        for child_name, trait in sorted(component.attributes.items(), key=_traitSortingKey):
            if not is_component_trait(trait):
                continue
            make_tree_items(getattr(component, child_name), child_name, depth + 1, newItem,
                            toggle_trait=trait.metadata.get("toggle"))

    win = QtWidgets.QWidget()
    win.setWindowTitle(getattr(component, "title", "Teval"))
    tree = QtWidgets.QTreeWidget(win)
    tree.setColumnCount(1)
    tree.setHeaderHidden(True)
    make_tree_items(component, "", 0, tree.invisibleRootItem())

    windowLayout = QtWidgets.QHBoxLayout(win)
    vSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, win)
    windowLayout.addWidget(vSplitter)

    splitter = QtWidgets.QSplitter()
    splitter.setChildrenCollapsible(False)
    vSplitter.addWidget(splitter)

    splitter.addWidget(tree)
    splitter.addWidget(stack)
    tree.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                       QtWidgets.QSizePolicy.Minimum)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)

    tree.itemClicked.connect(lambda x: stack.setCurrentIndex(x.widgetId))

    messagePane = QtWidgets.QGroupBox("Messages", win)
    vSplitter.addWidget(messagePane)

    msgPaneLayout = QtWidgets.QVBoxLayout(messagePane)
    msgBrowser = QtWidgets.QTextBrowser(messagePane)
    msgPaneLayout.addWidget(msgBrowser)

    return win, msgBrowser


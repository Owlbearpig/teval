from PySide6 import QtWidgets, QtCore, QtGui
from traitlets import Integer, Float, Unicode, Bool, Tuple, Enum
from common.measurement_selection import MeasurementSelection
from qtui.changeindicatorspinbox import ChangeIndicatorSpinBox
from qtui.changeindicatorlineedit import ChangeIndicatorLineEdit
from common.components import ComponentBase
from traitlets import Instance
from common.traits import Quantity, Path as PathTrait, ValueRange, MultiPathSelection, MultiPathClass
from pathlib import Path
import types
import logging
from collections import OrderedDict
from itertools import chain
import numpy as np
from qtui.flowlayout import FlowLayout
from common.traits import Q_, QuantityDict
from qtui.mplcanvas import MPLCanvas


def is_component_trait(x):
    return (isinstance(x, Instance) and issubclass(x.klass, ComponentBase))


def create_range_entry(component, name, trait):
    range_val = trait.get(component)
    sb_cnt = len(range_val)
    inner_value = range_val[0]

    is_integer = isinstance(inner_value, int)
    is_float = isinstance(inner_value, float)
    is_quantity = isinstance(inner_value, Q_)
    is_double_spinbox = not is_integer

    if is_integer:
        min_val = -2147483648 if trait.min is None else trait.min
        max_val = 2147483647 if trait.max is None else trait.max
    elif is_float:
        min_val = float('-inf') if trait.min is None else trait.min
        max_val = float('inf') if trait.max is None else trait.max
    elif is_quantity:
        min_val = float('-inf') if trait.min is None else trait.min.magnitude
        max_val = float('inf') if trait.max is None else trait.max.magnitude

    has_limits = not (np.isinf(min_val) or np.isinf(max_val))
    layout = QtWidgets.QHBoxLayout()

    spinboxes = []
    def setup_single_spinbox(sb_idx):
        def get_value():
            range_ = trait.get(component)
            return range_[sb_idx].magnitude if is_quantity else range_[sb_idx]

        spinbox = ChangeIndicatorSpinBox(is_double_spinbox=is_double_spinbox,
                                         actual_value_getter=get_value)
        spinboxes.append(spinbox)
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setToolTip(trait.help)
        spinbox.setReadOnly(trait.read_only)

        if trait.read_only:
            spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

        units = None
        if is_quantity:
            trait_val = trait.get(component)
            units = trait.metadata.get('preferred_units', None) or trait_val[sb_idx].units
            spinbox.setSuffix(f" {units:C~}")

        if is_double_spinbox and not has_limits:
            def sizeHint(self):
                original_hint = QtWidgets.QDoubleSpinBox.sizeHint(self)
                decimals = spinbox.decimals()
                font_metrics = spinbox.fontMetrics()
                text_width = font_metrics.horizontalAdvance(f"{get_value():.{decimals}f}")
                button_padding = 30
                suffix_padding = font_metrics.horizontalAdvance(spinbox.suffix())
                return QtCore.QSize(max(text_width + button_padding + suffix_padding, 20), original_hint.height())

            spinbox.sizeHint = types.MethodType(sizeHint, spinbox)
            spinbox.updateGeometry()

        layout.addWidget(spinbox)

        if not trait.read_only:
            apply = QtWidgets.QToolButton()
            apply.setFocusPolicy(QtCore.Qt.NoFocus)
            apply.setText('✓')
            apply.setAutoRaise(True)
            layout.addWidget(apply)

            def apply_value_to_component():
                new_range_val = list(trait.get(component))
                new_range_val[sb_idx] = spinbox.value() * units if is_quantity else spinbox.value()

                setattr(component, name, new_range_val)

            apply.clicked.connect(apply_value_to_component)
            apply.clicked.connect(spinbox.check_changed)
            spinbox.editingFinished.connect(apply_value_to_component)
            spinbox.editingFinished.connect(spinbox.check_changed)

        def update_spinbox_from_trait(new_val):
            spinbox.blockSignals(True)
            spinbox.setValue(new_val[sb_idx].to(units).magnitude if is_quantity else new_val[sb_idx])
            spinbox.blockSignals(False)

        if not trait.read_only:
            update_spinbox_from_trait(trait.get(component))
            component.observe(lambda c: update_spinbox_from_trait(c['new']), name)

    for i in range(2):
        setup_single_spinbox(sb_idx=i)

        if i != sb_cnt-1:
            separator_label = QtWidgets.QLabel("-")
            separator_label.setAlignment(QtCore.Qt.AlignCenter)
            separator_label.setStyleSheet("padding: 0 4px;")
            layout.addWidget(separator_label)

    layout.setContentsMargins(0, 0, 0, 0)
    layout.setStretch(0, 1)
    layout.setStretch(1, 0)

    return layout

def create_spinbox_entry(component, name, trait):
    is_integer = isinstance(trait, Integer)
    is_float = isinstance(trait, Float)
    is_quantity = isinstance(trait, Quantity)

    is_double_spinbox = not is_integer

    def get_value_with_units():
        return trait.get(component).magnitude

    def get_value_without_units():
        return trait.get(component)

    get_value = (get_value_with_units if is_quantity
                 else get_value_without_units)
    layout = QtWidgets.QHBoxLayout()
    spinbox = ChangeIndicatorSpinBox(is_double_spinbox=is_double_spinbox,
                                     actual_value_getter=get_value)
    spinbox.setToolTip(trait.help)

    if is_integer:
        spinbox.setMinimum(-2147483648 if trait.min is None else trait.min)
        spinbox.setMaximum(2147483647 if trait.max is None else trait.max)
    elif is_float:
        spinbox.setMinimum(float('-inf') if trait.min is None
                           else trait.min)
        spinbox.setMaximum(float('inf') if trait.max is None
                           else trait.max)
    elif is_quantity:
        spinbox.setMinimum(float('-inf') if trait.min is None
                           else trait.min.magnitude)
        spinbox.setMaximum(float('inf') if trait.max is None
                           else trait.max.magnitude)
    has_limits = not (np.isinf(spinbox.minimum()) or np.isinf(spinbox.maximum()))

    spinbox.setReadOnly(trait.read_only)
    if trait.read_only:
        spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

    if is_quantity:
        units = (trait.metadata.get('preferred_units', None) or
                 trait.get(component).units)
        spinbox.setSuffix(" {:C~}".format(units))

    if is_double_spinbox and not has_limits:
        def sizeHint(self):
            original_hint = QtWidgets.QDoubleSpinBox.sizeHint(self)
            decimals = spinbox.decimals()
            font_metrics = spinbox.fontMetrics()
            text_width = font_metrics.horizontalAdvance(str(f"{get_value():.{decimals}f}"))
            button_padding = 30
            suffix_padding = font_metrics.horizontalAdvance(spinbox.suffix())

            new_width = text_width + button_padding + suffix_padding

            return QtCore.QSize(max(new_width, 20), original_hint.height())

        spinbox.sizeHint = types.MethodType(sizeHint, spinbox)
        spinbox.updateGeometry()

    layout.addWidget(spinbox)
    if not trait.read_only:
        apply = QtWidgets.QToolButton()
        apply.setFocusPolicy(QtCore.Qt.NoFocus)
        apply.setText('✓')
        apply.setAutoRaise(True)
        layout.addWidget(apply)

    layout.setContentsMargins(0, 0, 0, 0)
    layout.setStretch(0, 1)
    layout.setStretch(1, 0)

    def apply_value_to_component_with_units():
        val = spinbox.value() * units
        setattr(component, name, val)

    def apply_value_to_component_without_units():
        setattr(component, name, spinbox.value())

    apply_value_to_component = \
        (apply_value_to_component_with_units if is_quantity
         else apply_value_to_component_without_units)

    def apply_value_to_spinbox_with_units(val):
        spinbox.blockSignals(True)
        spinbox.setValue(val.to(units).magnitude)
        spinbox.blockSignals(False)

    def apply_value_to_spinbox_without_units(val):
        spinbox.blockSignals(True)
        spinbox.setValue(val)
        spinbox.blockSignals(False)

    apply_value_to_spinbox = \
        (apply_value_to_spinbox_with_units if is_quantity
         else apply_value_to_spinbox_without_units)

    apply_value_to_spinbox(trait.get(component))
    component.observe(lambda c: apply_value_to_spinbox(c['new']), name)

    if not trait.read_only:
        apply.clicked.connect(apply_value_to_component)
        apply.clicked.connect(spinbox.check_changed)
        spinbox.editingFinished.connect(apply_value_to_component)
        spinbox.editingFinished.connect(spinbox.check_changed)

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


def create_checkbox(component, name, prettyName, trait):
    checkbox = QtWidgets.QCheckBox(prettyName)
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
    for item in trait.values:
        combobox.addItem(item.name, item)

    combobox.setCurrentText(trait.get(component).name)
    combobox.setToolTip(trait.help)

    component.observe(lambda change:
                      combobox.setCurrentText(change['new'].name), name)

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
        name = None
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
    initial_path = str(root_path) if root_path is not None else QtCore.QDir.homePath()
    model.setRootPath(initial_path)

    shown_filenames = getattr(trait.get(component), "shown_filenames", None)
    if shown_filenames:
        model.setNameFilters(shown_filenames)
        # model.setNameFilterDisables(False)
    else:
        model.setNameFilters(["*.txt"])

    tree = QtWidgets.QTreeView()
    tree.setModel(model)
    tree.hideColumn(1)  # Size
    tree.hideColumn(2)  # Type
    tree.hideColumn(3)  # Date Modified
    tree.setRootIndex(model.index(initial_path))
    tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    tree.setMinimumHeight(180)

    def update_trait_selection():
        selected_indexes = tree.selectionModel().selectedRows(column=0)
        paths = [Path(model.filePath(idx)) for idx in selected_indexes]
        new_root_path = Path(model.rootPath())
        multi_path_class = MultiPathClass(new_root_path, paths, shown_filenames=shown_filenames)
        setattr(component, name, multi_path_class)
        tree.setRootIndex(model.index(str(new_root_path)))

    tree.selectionModel().selectionChanged.connect(lambda *args: update_trait_selection())

    def update_root(change):
        new_val = change["new"]
        new_path_str = str(getattr(new_val, "root_path"))
        new_filenames = getattr(new_val, "shown_filenames", None)
        if new_filenames:
            model.setNameFilters(new_filenames)
            # model.setNameFilterDisables(False)

        model.setRootPath(new_path_str)
        tree.setRootIndex(model.index(new_path_str))

    component.observe(update_root, name)

    layout.addWidget(tree)
    return container

def create_plot_area(component, name, prettyName, trait):
    def draw(change):
        canvas.set_canvas_values(change["new"],
                                 trait.metadata.get("axes_labels", None),
                                 trait.metadata.get("data_label", None))

    canvas = MPLCanvas()

    component.observe(draw, name)

    canvas.setTitle(prettyName)

    initial_value = trait.get(component)
    if initial_value:
        canvas.set_canvas_values(initial_value,
                                 trait.metadata.get("axes_labels", None),
                                 trait.metadata.get("data_label", None))

    return canvas

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
        if isinstance(trait, QuantityDict):
            hasPlots = True
            continue

        group = _group(trait)

        if group not in groups:
            box = QtWidgets.QGroupBox(group, controlWidget)
            QtWidgets.QFormLayout(box)
            groups[group] = box

    controlWidget.param_widgets = {}
    for name, trait in traits:
        if isinstance(trait, QuantityDict):
            continue

        prettyName = _prettyName(trait, name)
        group = _group(trait)
        layout = groups[group].layout()
        if trait.metadata.get("fullwidth", False):
            groups[group].fullwidth = True

        groups[group].combine = trait.metadata.get("combine", False)

        field_widget = None
        if (isinstance(trait, ValueRange)):
            field_widget = create_range_entry(component, name, trait)
        elif (isinstance(trait, Quantity)):
            field_widget = create_spinbox_entry(component, name, trait)
        elif (isinstance(trait, Integer)):
            field_widget = create_spinbox_entry(component, name, trait)
        elif isinstance(trait, Enum) and not trait.read_only:
            field_widget = create_combobox(component, name, trait)
        elif isinstance(trait, Float):
            if trait.read_only and not (np.isinf(trait.min) or np.isinf(trait.max)):
                field_widget = create_progressbar(component, name, trait)
            else:
                field_widget = create_spinbox_entry(component, name, trait)
        elif isinstance(trait, Bool):
            field_widget = create_checkbox(component, name, prettyName, trait)
        elif isinstance(trait, Unicode):
            if trait.read_only:
                field_widget = create_label(component, name, trait)
            else:
                field_widget = create_lineedit(component, name, trait)
        elif isinstance(trait, PathTrait):
                field_widget = create_path_selector(component, name, prettyName, trait)
        elif isinstance(trait, MultiPathSelection):
            field_widget = create_tree_path_selector(component, name, prettyName, trait)

        if field_widget:
            if isinstance(trait, MultiPathSelection):
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

    if isinstance(component, MeasurementSelection):
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
        if not isinstance(trait, QuantityDict):
            continue
        prettyName = _prettyName(trait, name)

        plotBox.addWidget(create_plot_area(component, name, prettyName, trait))

    hSplitter = QtWidgets.QSplitter()
    hSplitter.addWidget(plotWidget)
    hSplitter.addWidget(scrollArea)
    hSplitter.setStretchFactor(1, 0)
    hSplitter.setStretchFactor(0, 1)
    hSplitter.setChildrenCollapsible(False)

    return hSplitter

def generate_ui(component):
    stack = QtWidgets.QStackedWidget()

    def make_tree_items(component, name, depth, treeitem):
        prettyName = component.object_name or name
        newItem = QtWidgets.QTreeWidgetItem(treeitem)
        newItem.setText(0, prettyName)
        if "AppRoot" in prettyName:
            newItem.setExpanded(True)

        widget = generate_component_ui(prettyName, component)
        newItem.widgetId = stack.addWidget(widget)

        for name, trait in sorted(component.attributes.items(), key=lambda x: x[0]):
            if not is_component_trait(trait):
                continue
            cInst = getattr(component, name)
            make_tree_items(cInst, name, depth + 1, newItem)

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


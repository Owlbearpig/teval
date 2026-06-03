from PySide6 import QtWidgets, QtCore


from PyQt5 import QtWidgets, QtCore
from .changeindicatorspinbox import ChangeIndicatorSpinBox
from .changeindicatorlineedit import ChangeIndicatorLineEdit
from .flowlayout import FlowLayout

try:
    from .pyqtgraphplotter import PyQtGraphPlotter
    usePyQtGraph = True
except:
    from .mplcanvas import MPLCanvas
    usePyQtGraph = False

import asyncio
from common import ComponentBase
from common.traits import DataSet as DataSetTrait
from traitlets import Instance, Float, Bool, Integer, Enum, Unicode
from collections import OrderedDict
from itertools import chain
from common.traits import Quantity, Path as PathTrait
from pathlib import Path
import logging
import numpy as np



def run_action(func):
    ret = func()
    if asyncio.iscoroutine(ret):
        asyncio.ensure_future(ret)


def is_component_trait(x):
    return (isinstance(x, Instance) and issubclass(x.klass, ComponentBase))


def create_spinbox_entry(component, name, trait):
    is_integer = isinstance(trait, Integer)
    is_float = isinstance(trait, Float)
    is_quantity = isinstance(trait, Quantity)

    def get_value_with_units():
        return trait.get(component).magnitude

    def get_value_without_units():
        return trait.get(component)

    get_value = (get_value_with_units if is_quantity
                 else get_value_without_units)

    layout = QtWidgets.QHBoxLayout()
    spinbox = ChangeIndicatorSpinBox(is_double_spinbox=not is_integer,
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

    spinbox.setReadOnly(trait.read_only)
    if trait.read_only:
        spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

    if is_quantity:
        units = (trait.metadata.get('preferred_units', None) or
                 trait.get(component).units)
        spinbox.setSuffix(" {:C~}".format(units))

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
    qaction = QtWidgets.QAction(action.metadata.get('name', action.__name__),
                                None)
    qaction.setToolTip(action.help)

    qaction.triggered.connect(lambda: run_action(action))

    return qaction


def create_plot_area(component, name, prettyName, trait):
    if usePyQtGraph:
        def draw(change):
            canvas.drawDataSet(change['new'])
    else:
        def draw(change):
            canvas.dataIsPower = trait.metadata.get('is_power', False)
            canvas.drawDataSet(change['new'],
                               trait.metadata.get('axes_labels', None),
                               trait.metadata.get('data_label', None))

    if usePyQtGraph:
        canvas = PyQtGraphPlotter()
        canvas.setLabels(trait.metadata.get('axes_labels', None),
                         trait.metadata.get('data_label', None))
        canvas.dataIsPower = trait.metadata.get('is_power', False)
    else:
        canvas = MPLCanvas()

    component.observe(draw, name)
    canvas.setTitle(prettyName)

    return canvas


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
    label.setText(trait.get(component))
    label.setToolTip(trait.help)

    component.observe(lambda change: label.setText(change['new']), name)

    return label


def create_lineedit(component, name, trait):
    lineEdit = ChangeIndicatorLineEdit(actual_value_getter=
                                       lambda: trait.get(component))
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

    layout.addWidget(lineEdit)
    layout.addWidget(choose)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setStretch(0, 1)
    layout.setStretch(1, 0)

    return layout


def _group(trait):
    return trait.metadata.get('group', 'General')


def _prettyName(trait, name):
    return trait.metadata.get('name', name)


traitPriority = {
    'Unicode': -1,
    'Path': 0,
    'Float': 1,
    'Int': 1,
    'Quantity': 1,
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

    userPrio = trait.metadata.get('priority', 999)

    return prio, userPrio, name


def generate_component_ui(name, component):
    controlWidget = QtWidgets.QWidget()

    # filter and sort traits
    traits = [(name, trait) for name, trait
              in sorted(chain(component.traits().items(),
                              component.actions), key=_traitSortingKey)
              if not is_component_trait(trait)]

    # pre-create group boxes
    groups = OrderedDict()

    hasPlots = False
    for name, trait in traits:
        if isinstance(trait, DataSetTrait):
            hasPlots = True
            continue

        group = _group(trait)

        if group not in groups:
            box = QtWidgets.QGroupBox(group, controlWidget)
            QtWidgets.QFormLayout(box)
            groups[group] = box

    for name, trait in traits:
        if isinstance(trait, DataSetTrait):
            continue

        prettyName = _prettyName(trait, name)
        group = _group(trait)
        layout = groups[group].layout()

        if (isinstance(trait, Quantity)):
            layout.addRow(prettyName + ": ",
                          create_spinbox_entry(component, name, trait))
        if (isinstance(trait, Integer)):
            layout.addRow(prettyName + ": ",
                          create_spinbox_entry(component, name, trait))
        elif isinstance(trait, Enum) and not trait.read_only:
            layout.addRow(prettyName + ": ",
                          create_combobox(component, name, trait))
        elif isinstance(trait, Float):
            if trait.read_only and not (np.isinf(trait.min) or
                                        np.isinf(trait.max)):
                layout.addRow(prettyName + ": ",
                              create_progressbar(component, name, trait))
            else:
                layout.addRow(prettyName + ": ",
                              create_spinbox_entry(component, name, trait))
        elif isinstance(trait, Bool):
            layout.addRow(" ",
                          create_checkbox(component, name, prettyName, trait))
        elif isinstance(trait, Unicode):
            if trait.read_only:
                layout.addRow(prettyName + ": ",
                              create_label(component, name, trait))
            else:
                layout.addRow(prettyName + ": ",
                              create_lineedit(component, name, trait))
        elif isinstance(trait, PathTrait):
                layout.addRow(prettyName + ": ",
                              create_path_selector(component, name, prettyName,
                                                   trait))
        elif callable(trait):
            qaction = create_action(component, trait)
            qaction.setParent(controlWidget)
            btn = QtWidgets.QToolButton()
            btn.setDefaultAction(qaction)
            layout.addRow(None, btn)

    controlLayout = FlowLayout(controlWidget)
    for i, group in enumerate(groups.values()):
        controlLayout.addWidget(group)

    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setFrameStyle(QtWidgets.QFrame.NoFrame)
    scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
    scrollArea.setWidgetResizable(True)

    scrollArea.setWidget(controlWidget)

    scrollArea.setMinimumWidth(scrollArea.sizeHint().width())

    if not groups:
        scrollArea.hide()

    if not hasPlots:
        return scrollArea

    plotWidget = QtWidgets.QWidget()
    plotBox = QtWidgets.QVBoxLayout(plotWidget)
    plotBox.setContentsMargins(0, 0, 0, 0)

    for name, trait in traits:
        if not isinstance(trait, DataSetTrait):
            continue
        prettyName = _prettyName(trait, name)

        plotBox.addWidget(create_plot_area(component, name, prettyName, trait))

    splitter = QtWidgets.QSplitter()
    splitter.addWidget(plotWidget)
    splitter.addWidget(scrollArea)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 0)
    splitter.setChildrenCollapsible(False)

    return splitter



def generate_ui(component):

    stack = QtWidgets.QStackedWidget()

    def make_tree_items(component, name, depth, treeitem):

        prettyName = component.objectName or name
        newItem = QtWidgets.QTreeWidgetItem(treeitem)
        newItem.setText(0, prettyName)
        newItem.setExpanded(True)

        widget = generate_component_ui(prettyName, component)
        newItem.widgetId = stack.addWidget(widget)

        for name, trait in sorted(component.attributes.items(),
                                  key=lambda x: x[0]):
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

    # tree.itemClicked.connect(lambda x: stack.setCurrentIndex(x.widgetId))

    vSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, win)
    messagePane = QtWidgets.QGroupBox("Messages", win)
    vSplitter.addWidget(messagePane)

    msgPaneLayout = QtWidgets.QVBoxLayout(messagePane)
    msgBrowser = QtWidgets.QTextBrowser(messagePane)
    msgPaneLayout.addWidget(msgBrowser)

    return win, msgBrowser


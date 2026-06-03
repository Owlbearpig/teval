from PySide6 import QtWidgets, QtCore
from dataclasses import fields, is_dataclass
from enum import Enum
from common.components import ComponentBase


def is_component(x):
    return issubclass(x, ComponentBase)

def build_form_from_component(component, parent_layout, widget_map=None):
    if widget_map is None:
        widget_map = {}

    if not is_dataclass(component):
        return widget_map

    all_fields = list(fields(component))
    sorted_fields = sorted(
        all_fields,
        key=lambda f: f.metadata.get("priority", 0),
        reverse=True
    )

    for field in sorted_fields:
        field_name = field.name
        field_value = getattr(component, field_name)
        field_type = field.type
        label_text = field_name.replace('_', ' ').title()

        is_readonly = field.metadata.get("readonly", False)

        if is_dataclass(field_value):
            group_box = QtWidgets.QGroupBox(label_text)
            group_layout = QtWidgets.QVBoxLayout(group_box)
            widget_map[field_name] = {}
            build_form_from_component(field_value, group_layout, widget_map[field_name])
            parent_layout.addWidget(group_box)

            if is_readonly:
                group_box.setEnabled(False)
            continue

        row_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(label_text)
        row_layout.addWidget(label)

        if isinstance(field_value, Enum) or (isinstance(field_type, type) and issubclass(field_type, Enum)):
            widget = QtWidgets.QComboBox()

            if isinstance(field_type, type) and issubclass(field_type, Enum):
                enum_class = field_type
            else:
                enum_class = field_value.__class__

            widget.addItems([e.name for e in enum_class])

            if isinstance(field_value, Enum):
                widget.setCurrentText(field_value.name)

            if is_readonly:
                widget.setEnabled(False)

        elif field_type is bool or isinstance(field_value, bool):
            widget = QtWidgets.QCheckBox(label_text)
            widget.setChecked(field_value)
            label.setText("")

            if is_readonly:
                widget.setEnabled(False)

        elif field_type is int or isinstance(field_value, int):
            widget = QtWidgets.QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(field_value)

            if is_readonly:
                widget.setReadOnly(True)

        elif field_type is float or isinstance(field_value, float):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setValue(field_value)

            if is_readonly:
                widget.setReadOnly(True)

        else:
            widget = QtWidgets.QLineEdit()
            widget.setText(str(field_value if field_value is not None else ""))

            if is_readonly:
                widget.setReadOnly(True)

        row_layout.addWidget(widget, stretch=1)
        parent_layout.addLayout(row_layout)

        widget_map[field_name] = widget

    return widget_map
    # return widget

def generate_ui(component):
    stack = QtWidgets.QStackedWidget()

    def make_tree_items(component, name, depth, treeitem):
        prettyName = component.object_name or name
        newItem = QtWidgets.QTreeWidgetItem(treeitem)
        newItem.setText(0, prettyName)
        newItem.setExpanded(True)

        # widget = build_form_from_component(prettyName, component)
        widget = QtWidgets.QGroupBox(prettyName, win)
        newItem.widgetId = stack.addWidget(widget)

        for name, obj in component.__dict__.items():
            if not isinstance(obj, ComponentBase):
                continue

            make_tree_items(obj, name, depth + 1, newItem)

            """
            if not is_component(trait):
                continue
            cInst = getattr(component, name)
            make_tree_items(cInst, name, depth + 1, newItem)
            """

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

    """
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    form_widget = QtWidgets.QWidget()
    form_layout = QtWidgets.QVBoxLayout(form_widget)
    """

    splitter.addWidget(tree)
    splitter.addWidget(stack)
    tree.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                       QtWidgets.QSizePolicy.Minimum)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)

    tree.itemClicked.connect(lambda x: stack.setCurrentIndex(x.widgetId))

    # build_form_from_component(component.settings, form_layout)

    # form_layout.addStretch()
    # scroll_area.setWidget(form_widget)
    # windowLayout.addWidget(scroll_area, stretch=3)

    # log_group = QtWidgets.QGroupBox("Application Logs")
    # log_layout = QtWidgets.QVBoxLayout(log_group)
    # msg_browser = QtWidgets.QTextBrowser()
    # log_layout.addWidget(msg_browser)

    # windowLayout.addWidget(log_group, stretch=2)

    messagePane = QtWidgets.QGroupBox("Messages", win)
    vSplitter.addWidget(messagePane)

    msgPaneLayout = QtWidgets.QVBoxLayout(messagePane)
    msgBrowser = QtWidgets.QTextBrowser(messagePane)
    msgPaneLayout.addWidget(msgBrowser)

    return win, msgBrowser


from PySide6 import QtWidgets, QtCore
from dataclasses import fields, is_dataclass
from enum import Enum

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

            # SpinBoxes support setReadOnly (allows selecting/copying text, but no changing)
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

            # LineEdits support setReadOnly perfectly
            if is_readonly:
                widget.setReadOnly(True)

        row_layout.addWidget(widget, stretch=1)
        parent_layout.addLayout(row_layout)

        widget_map[field_name] = widget

    return widget_map

def generate_ui(component):
    """
    Dynamically builds a UI window by inspecting a dataclass instance.
    Returns: (Main Widget, QTextBrowser for logs)
    """
    main_window = QtWidgets.QWidget()
    main_layout = QtWidgets.QHBoxLayout(main_window)

    # Left side: Form scroll area for the settings variables
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    form_widget = QtWidgets.QWidget()
    form_layout = QtWidgets.QVBoxLayout(form_widget)

    # Dynamically build inputs for the AppSettings (or any subclass) object
    build_form_from_component(component.settings, form_layout)

    # Push everything up and set the scroll widget
    form_layout.addStretch()
    scroll_area.setWidget(form_widget)
    main_layout.addWidget(scroll_area, stretch=3)

    # Right side: The log window (QTextBrowser)
    log_group = QtWidgets.QGroupBox("Application Logs")
    log_layout = QtWidgets.QVBoxLayout(log_group)
    msg_browser = QtWidgets.QTextBrowser()
    log_layout.addWidget(msg_browser)

    main_layout.addWidget(log_group, stretch=2)

    # Set window title using dynamic attribute fallback
    main_window.setWindowTitle(getattr(component, "title", "Configuration Panel"))

    return main_window, msg_browser


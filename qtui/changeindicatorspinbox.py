from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox
from PySide6.QtGui import QPalette


def ChangeIndicatorSpinBox(*args, actual_value_getter,
                           is_double_spinbox=False, **kwargs):
    if is_double_spinbox:
        spinbox = QDoubleSpinBox(*args, **kwargs)
        spinbox.setDecimals(3)
    else:
        spinbox = QSpinBox(*args, **kwargs)

    spinbox.unchanged_palette = spinbox.palette()

    spinbox.changed_palette = spinbox.palette()
    highlightColor = spinbox.changed_palette.color(QPalette.Highlight)
    highlightColor.setHsl(0xFF - highlightColor.hslHue(),
                          highlightColor.hslSaturation(),
                          highlightColor.lightness())

    spinbox.changed_palette.setColor(QPalette.Base, highlightColor)

    def check_changed():
        actualValue = actual_value_getter()
        if spinbox.value() != actualValue:
            spinbox.setPalette(spinbox.changed_palette)
        else:
            spinbox.setPalette(spinbox.unchanged_palette)

    spinbox.check_changed = check_changed
    spinbox.valueChanged.connect(spinbox.check_changed)

    return spinbox

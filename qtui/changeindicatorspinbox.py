from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox
from PySide6.QtGui import QPalette, QValidator
from PySide6.QtCore import QRegularExpression


class ScientificDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._regex = QRegularExpression(r"^[-+]?[0-9]*[,]?[0-9]*([eE][-+]?[0-9]*)?$")

    def validate(self, text, pos):
        clean_text = text.replace(self.suffix(), "").strip()

        match = self._regex.match(clean_text)

        if match.hasMatch():
            return QValidator.State.Acceptable, text, pos

        return QValidator.State.Invalid, text, pos

def ChangeIndicatorSpinBox(*args, actual_value_getter,
                           is_double_spinbox=False, **kwargs):
    sb_decimals = 32
    if is_double_spinbox:
        spinbox = ScientificDoubleSpinBox(*args, **kwargs)
        spinbox.setDecimals(sb_decimals)
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
        below_prec = actualValue < 10**-sb_decimals and spinbox.value() < 10**-sb_decimals
        if spinbox.value() != actualValue and not below_prec:
            spinbox.setPalette(spinbox.changed_palette)
        else:
            spinbox.setPalette(spinbox.unchanged_palette)

    spinbox.check_changed = check_changed
    spinbox.valueChanged.connect(spinbox.check_changed)

    return spinbox

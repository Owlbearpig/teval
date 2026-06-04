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

from PySide6.QtWidgets import QLineEdit
from PySide6.QtGui import QPalette


def ChangeIndicatorLineEdit(*args, actual_value_getter, **kwargs):
    lineEdit = QLineEdit(*args, **kwargs)

    lineEdit.unchanged_palette = lineEdit.palette()

    lineEdit.changed_palette = lineEdit.palette()
    highlightColor = lineEdit.changed_palette.color(QPalette.Highlight)
    highlightColor.setHsl(0xFF - highlightColor.hslHue(),
                          highlightColor.hslSaturation(),
                          highlightColor.lightness())

    lineEdit.changed_palette.setColor(QPalette.Base, highlightColor)

    def check_changed():
        actualValue = actual_value_getter()
        if lineEdit.text() != actualValue:
            lineEdit.setPalette(lineEdit.changed_palette)
        else:
            lineEdit.setPalette(lineEdit.unchanged_palette)

    lineEdit.check_changed = check_changed
    lineEdit.textChanged.connect(lineEdit.check_changed)

    return lineEdit
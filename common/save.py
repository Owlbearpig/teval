# -*- coding: utf-8 -*-
"""
This file is part of Taipan.

Copyright (C) 2015 - 2017 Arno Rehn <arno@arnorehn.de>

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
from common.components import ComponentBase
from common.traits import Path as PathTrait
from enum import Enum, unique
from traitlets import Bool, Enum as EnumTrait, Unicode
import numpy as np
from datetime import datetime
import logging
from copy import deepcopy
from common.consts import result_dir
from pathlib import Path

test_result = {
    # --- Scalars ---
    "d_um": 10.03,
    "q_val": 0.0035,
    "epochs": 150,
    "converged": True,

    # --- Strings ---
    "model_name": "ResNet-50_Eval",
    "timestamp": "2026-06-29_12:35:00",

    # --- 1D NumPy Arrays ---
    "losses": np.array([0.69, 0.45, 0.31, 0.18, 0.09]),
    "accuracies": np.array([0.55, 0.72, 0.84, 0.91, 0.96]),
    "freq_axis": np.ones(4001),
    "n_sub": np.ones(4001)*3.1415,

    # --- Standard Python Collections ---
    "class_labels": ["cat", "dog", "frog"],
}


def _getManipulatorValueInPreferredUnits(m):
    val = m.value

    pref_units = m.trait_metadata('value', 'preferred_units')
    if pref_units:
        val = val.to(pref_units)

    return val

class ResultSaver(ComponentBase):

    # base_path = PathTrait(default_value=result_dir, is_file=False, must_exist=False).tag(name="Path")
    base_path = Path(result_dir)

    textFileWithHeaders = Bool(False).tag(name="Write header to text files")
    fileNameTemplate = Unicode('{date}-{name}',
                               help="File name template, valid identifiers "
                                    "are:\n"
                                    "{name}: The main file name\n"
                                    "{date}: The current date and time").tag(
                               name="File name template")
    mainFileName = Unicode('data').tag(name="Main file name")

    enabled = Bool(False, help="Whether data storage is enabled").tag(
                         name="Enabled")

    _manipulators = {}
    _attributes = {}

    # from https://msdn.microsoft.com/en-us/library/aa365247
    _forbiddenCharacters = r'"*/:<>?\|'
    _fileNameTranslationTable = str.maketrans(_forbiddenCharacters,
                                              '_' * len(_forbiddenCharacters))

    def registerManipulator(self, manipulator, name=None):
        if name is None:
            name = manipulator.objectName

        self._manipulators[name] = manipulator

        trait = deepcopy(self.traits()['fileNameTemplate'])
        additionalHelpString = ('\n{{{}}}: The value of manipulator {}'
                                .format(name, manipulator.objectName))
        trait.help += additionalHelpString
        if 'help' in trait.metadata:
            trait.metadata['help'] += additionalHelpString
        self.add_traits(fileNameTemplate=trait)

    def registerObjectAttribute(self, inst, attr, name=None):
        if name is None:
            name = attr

        self._attributes[name] = (inst, attr)

        trait = deepcopy(self.traits()['fileNameTemplate'])
        additionalHelpString = ('\n{{{}}}: The value of "{}.{}"'
                                .format(name, str(inst), attr))
        trait.help += additionalHelpString
        if 'help' in trait.metadata:
            trait.metadata['help'] += additionalHelpString
        self.add_traits(fileNameTemplate=trait)


    def _getFileName(self):

        self.base_path /= Path(self.script_name).stem
        self.base_path.mkdir(parents=True, exist_ok=True)

        date = datetime.now().isoformat().replace(':', '-')

        manipValues = {k: '{:.3fC~}'
                       .format(_getManipulatorValueInPreferredUnits(m))
                       for k, m in self._manipulators.items()}

        attributeValues = {k: str(getattr(inst, name))
                           for k, (inst, name) in self._attributes.items()}

        formattedName = self.fileNameTemplate.format(date=date,
                                                     name=self.mainFileName,
                                                     **manipValues,
                                                     **attributeValues)
        formattedName = formattedName.translate(self._fileNameTranslationTable)

        return str(self.base_path.joinpath(formattedName))


    def _saveNumpy(self, data):
        fileName = self._getFileName()
        print(fileName)
        return
        axesUnits = ['{:C}'.format(ax.units) for ax in data.axes]
        dataUnits = '{:C}'.format(data.data.units)
        unitlessAxes = [ax.magnitude for ax in data.axes]
        np.savez_compressed(fileName, axes=unitlessAxes, axesUnits=axesUnits,
                            data=data.data.magnitude, dataUnits=dataUnits)
        return fileName

    def process(self, data):
        if not self.enabled:
            logging.info("Data storage is disabled, not saving results.")
            return

        filename = self._saveNumpy(data)

        logging.info("Saved result as {}".format(filename))
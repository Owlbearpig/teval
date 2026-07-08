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
from common.eval_component.quantity_set import QuantityDict as QuantityDictClass

from common.units import Q_
from enum import Enum, unique
from traitlets import Bool, Enum as EnumTrait, Unicode
import numpy as np
from datetime import datetime
import logging
from copy import deepcopy
from common.consts import result_dir
from pathlib import Path


rnd_arr = np.random.random

test_result = {
    # --- Scalars ---
    "d": Q_(10.03, "µm"),
    "q_val": Q_(0.035, ""),
    "gof": Q_(0.0, ""),
    "shift": Q_(1.203, ""),
    "converged": True,

    # --- Strings ---
    # "timestamp": "2026-06-29_12:35:00",
    "timestamp": "2026-06-29_13:35:00000000000000000000000000000",

    # --- Datasets ( Q_(x) ) ---
    # "delta_n": Data# np.array([3*np.arange(1, 4002), rnd_arr(4001)]).T,
    "delta_alpha": np.array([3*np.arange(1, 4002), rnd_arr(4001)]).T,
    "n0": np.array([3*np.arange(1, 4002), rnd_arr(4001)]).T,
    "n": np.array([3*np.arange(1, 4002), rnd_arr(4001)]).T,
    "k": np.array([3*np.arange(1, 4002), rnd_arr(4001)]).T,
    "alpha": np.array([3*np.arange(1, 4002), rnd_arr(4001)]).T,
    "t_mod": np.array([3*np.arange(1, 4002), rnd_arr(4001)]).T,
    "sam_mod": np.array([3*np.arange(1, 4002), rnd_arr(4001)]).T,
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

    enabled = Bool(True, help="Whether data storage is enabled").tag(
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

        save_path = self.base_path / Path(self.script_name).stem
        save_path.mkdir(parents=True, exist_ok=True)

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

        return str(save_path.joinpath(formattedName))

    def _saveNumpy(self, eval_result):
        fileName = self._getFileName()

        attributes = {
            k: (v.magnitude if hasattr(v, "magnitude") else v)
            for k, v in eval_result.trait_values().items()
            if not isinstance(v, QuantityDictClass)
        }

        for key, dataset in eval_result.quantity_dict.items():
            data = dataset.data.magnitude if isinstance(dataset.data, Q_) else dataset.data
            axis = dataset.axes[0].magnitude if isinstance(dataset.axes[0], Q_) else dataset.axes[0]
            attributes[key] = np.array([axis, data]).T

        np.savez_compressed(fileName, **attributes, allow_pickle=False)

        return fileName

    def process(self, eval_result):
        if not self.enabled:
            logging.info("Data storage is disabled, not saving results.")
            return

        filename = self._saveNumpy(eval_result)

        logging.info("Saved result as {}".format(filename))
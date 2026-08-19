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
from common.eval_component.quantity_set import DataSetDict as QuantityDictClass, DataSet
from common.units import Q_
from enum import Enum, unique
from common.traits import ValueRange
from traitlets import Bool, Enum as EnumTrait, Unicode
import numpy as np
from datetime import datetime
import logging
from copy import deepcopy
from common.consts import result_dir
from pathlib import Path
import h5py

rnd_arr = np.random.random

freq_axis = Q_(np.linspace(1, 10, 4001), "THz")
test_result = {
    # --- Scalars ---
    "d": Q_(10*np.random.random(), "µm"),
    "q_val": Q_(1e-3*np.random.random(), ""),
    "gof": Q_(1e-5*np.random.random(), ""),
    "shift": Q_(1e2*np.random.random(), "fs"),
    "converged": True,

    # --- Strings ---
    # "timestamp": "2026-06-29_12:35:00",
    "timestamp": str(datetime.now().isoformat()),

    # --- Datasets ( Q_(x) ) ---
    "delta_n": DataSet(axes=[freq_axis], data=Q_(rnd_arr(4001), "S"), axes_labels=["Frequency"]),
    "delta_alpha": DataSet(axes=[freq_axis], data=Q_(rnd_arr(4001), "m"), axes_labels=["Frequency"]),
    "n0": DataSet(axes=[freq_axis], data=Q_(rnd_arr(4001), "T"), data_label="Simple n", axes_labels=["Frequency"]),
    "n": DataSet(axes=[freq_axis], data=Q_(rnd_arr(4001), "nm"), axes_labels=["Frequency"]),
    "k": DataSet(axes=[freq_axis], data=Q_(rnd_arr(4001), "W"), axes_labels=["Frequency"]),
    "alpha": DataSet(axes=[freq_axis], data=Q_(rnd_arr(4001), "1/cm"), axes_labels=["Frequency"]),
    "t_mod": DataSet(axes=[freq_axis], data=Q_(rnd_arr(4001), "V"), axes_labels=["ABE"]),
    "sam_mod": DataSet(axes=[freq_axis], data=Q_(rnd_arr(4001), "J")),
}


def _getManipulatorValueInPreferredUnits(m):
    val = m.value

    pref_units = m.trait_metadata('value', 'preferred_units')
    if pref_units:
        val = val.to(pref_units)

    return val

class ResultSaver(ComponentBase):
    @unique
    class Formats(Enum):
        HDF5 = 0
        Numpy = 1

    extension = {Formats.HDF5: '.hdf5', Formats.Numpy: '.npz'}

    # base_path = PathTrait(default_value=result_dir, is_file=False, must_exist=False).tag(name="Path")
    base_path = Path(result_dir)
    fileFormat = EnumTrait(Formats, Formats.HDF5).tag(name="File format")

    textFileWithHeaders = Bool(False).tag(name="Write header to text files")
    fileNameTemplate = Unicode('{date}-{name}-{result_type}',
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

    def _format_attribute(self, inst, name):
        attr = getattr(inst, name)
        if isinstance(attr, list) and len(attr) == 2:
            if isinstance(attr[0], Q_):
                s = 'x{} {:C~}'.format(attr[0].magnitude, attr[0].units)
                s +='-y{} {:C~}'.format(attr[1].magnitude, attr[1].units)
            else:
                s = "x{}-y{}".format(*attr)
            return s
        else:
            return str(attr)

    def _getFileName(self):

        save_path = self.base_path / Path(self.script_name).stem
        save_path.mkdir(parents=True, exist_ok=True)

        date = datetime.now().isoformat().replace(':', '-')

        manipValues = {k: '{:.3fC~}'
                       .format(_getManipulatorValueInPreferredUnits(m))
                       for k, m in self._manipulators.items()}

        attributeValues = {k: self._format_attribute(inst, name)
                           for k, (inst, name) in self._attributes.items()}

        formattedName = self.fileNameTemplate.format(date=date,
                                                     name=self.mainFileName,
                                                     **manipValues,
                                                     **attributeValues)
        formattedName += self.extension[self.fileFormat]
        formattedName = formattedName.translate(self._fileNameTranslationTable)

        return str(save_path.joinpath(formattedName))

    def _saveNumpy(self, eval_result):
        fileName = self._getFileName()

        attributes = {}
        for k, v in eval_result.trait_values().items():
            if isinstance(v, Q_):
                attributes[f"{k}__QK__quantity_magnitude"] = v.magnitude
                attributes[f"{k}__QK__quantity_units"] = '{:C}'.format(v.units)
            elif isinstance(v, QuantityDictClass):
                for key, dataset in v.items():
                    attributes[f"{key}__DSK__data_units"] = '{:C}'.format(dataset.data.units)
                    attributes[f"{key}__DSK__data_magnitude"] = dataset.data.magnitude
                    for i, ax in enumerate(dataset.axes):
                        attributes[f"{key}__DSK__axes_units_{i}"] = '{:C}'.format(ax.units)
                        attributes[f"{key}__DSK__axes_magnitude_{i}"] = ax.magnitude
            else:
                attributes[k] = v

        np.savez_compressed(fileName, **attributes, allow_pickle=False)

        return fileName

    def _saveHDF5(self, eval_result):
        fileName = self._getFileName()

        with h5py.File(fileName, "w") as f:
            scalar_group = f.create_group("scalars")
            for k in eval_result.traits().keys():
                v = getattr(eval_result, k)

                if isinstance(v, QuantityDictClass):
                    continue
                elif isinstance(v, Q_):
                    dset = scalar_group.create_dataset(k, data=v.magnitude)
                    dset.attrs["unit"] = "{:C}".format(v.units)
                elif isinstance(v, Path):
                    dset = scalar_group.create_dataset(k, data=str(v))
                    dset.attrs["path_type"] = type(v).__name__
                else:
                    scalar_group[k] = v

            qd_group = f.create_group("quantity_dict")
            for key, dataset in eval_result.quantity_dict.items():
                prefix = str(key)
                dataset_group = qd_group.create_group(prefix)

                dset = dataset_group.create_dataset("data", data=dataset.data.magnitude)
                dset.attrs["unit"] = "{:C}".format(dataset.data.units)
                dset.attrs["data_label"] = dataset.data_label

                axes_group = dataset_group.create_group("axes")
                for i, ax in enumerate(dataset.axes):
                    ax_subgroup = axes_group.create_group(f"axis_{i}")
                    dset = ax_subgroup.create_dataset("axis_dset", data=ax.magnitude)
                    dset.attrs["unit"] = "{:C}".format(ax.units)
                    try:
                        dset.attrs["axis_label"] = dataset.axes_labels[i]
                    except IndexError:
                        dset.attrs["axis_label"] = ""

        return fileName

    def process(self, eval_result):
        if not self.enabled:
            logging.info("Data storage is disabled, not saving results.")
            return

        filename = None
        if self.fileFormat == self.Formats.HDF5:
            filename = self._saveHDF5(eval_result)
        elif self.fileFormat == self.Formats.Numpy:
            filename = self._saveNumpy(eval_result)

        logging.info("Saved result as {}".format(filename))
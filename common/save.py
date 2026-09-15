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
from common.eval_component.eval_result import EvalResult, EvalResultData
from common.eval_component.quantity_set import QuantityDataSetDict as QuantityDictClass, QuantityDataSet
from common.units import Q_
from enum import Enum, unique
from common.traits import ValueRange, Path as PathTrait
from traitlets import Bool, Enum as EnumTrait, Unicode
import numpy as np
from datetime import datetime
import logging
from copy import deepcopy
from common.consts import result_dir
from pathlib import Path
import h5py
import re

def _getManipulatorValueInPreferredUnits(m):
    val = m.value

    pref_units = m.trait_metadata('value', 'preferred_units')
    if pref_units:
        val = val.to(pref_units)

    return val

class ResultSaver(ComponentBase):

    base_path = PathTrait(default_value=result_dir, is_file=False, must_exist=False).tag(name="Path")

    textFileWithHeaders = Bool(False).tag(name="Write header to text files")
    fileNameTemplate = Unicode('{date}-{name}-{result_type}',
                               help="File name template, valid identifiers "
                                    "are:\n"
                                    "{name}: The main file name\n"
                                    "{date}: The current date and time").tag(
                               name="File name template")
    mainFileName = Unicode('data').tag(name="Main file name", fullwidth=True)

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

    def _get_main_file_name(self, meas_names=None):
        if not meas_names:
            return self.mainFileName
        main_fn = self.mainFileName
        if len(main_fn) < 2 or main_fn[:2] != "$$":
            return main_fn

        no_time_names = []
        for meas in meas_names:
            if len(meas) > 26:
                no_time_names.append(meas[26:])

        for meas_name in set(no_time_names):
            match = re.search(main_fn[2:], meas_name)
            if match:
                return match.group(0)

        return main_fn

    def _getFileName(self, meas_names):

        save_path = self.base_path
        save_path.mkdir(parents=True, exist_ok=True)

        date = datetime.now().isoformat().replace(':', '-')

        manipValues = {k: '{:.3fC~}'
                       .format(_getManipulatorValueInPreferredUnits(m))
                       for k, m in self._manipulators.items()}

        attributeValues = {k: self._format_attribute(inst, name)
                           for k, (inst, name) in self._attributes.items()}

        formattedName = self.fileNameTemplate.format(date=date,
                                                     name=self._get_main_file_name(meas_names),
                                                     **manipValues,
                                                     **attributeValues)
        formattedName += ".hdf5"
        formattedName = formattedName.translate(self._fileNameTranslationTable)

        return str(save_path.joinpath(formattedName))

    def _saveHDF5(self, eval_result: EvalResult):
        fileName = self._getFileName(eval_result.eval_result_data.measurement_names)
        eval_data: EvalResultData = eval_result.eval_result_data

        def write_quantity(group, name, quantity):
            if hasattr(quantity, "magnitude"):
                dset = group.create_dataset(name, data=quantity.magnitude)
                dset.attrs["unit"] = "{:C}".format(quantity.units)
            else:
                group.create_dataset(name, data=quantity)

        def quantity_dataset_to_hdf5group(q_dataset, hdf5_group):
            dset = hdf5_group.create_dataset("data", data=q_dataset.data.magnitude)
            dset.attrs["unit"] = "{:C}".format(q_dataset.data.units)
            dset.attrs["data_label"] = q_dataset.data_label or ""
            hdf5_group.create_dataset("uncert", data=q_dataset.uncert.magnitude)

            axes_group = hdf5_group.create_group("axes")
            for i, ax in enumerate(q_dataset.axes):
                ax_subgroup = axes_group.create_group(f"axis_{i}")
                dset_ax = ax_subgroup.create_dataset("axis_dset", data=ax.magnitude)
                dset_ax.attrs["unit"] = "{:C}".format(ax.units)
                try:
                    dset_ax.attrs["axis_label"] = q_dataset.axes_labels[i]
                except IndexError:
                    dset_ax.attrs["axis_label"] = ""

        with h5py.File(fileName, "w") as f:
            f.attrs["result_type"] = eval_data.result_type
            f.attrs["dataset_path"] = str(eval_data.dataset_path)
            f.attrs["model_name"] = eval_data.model_name
            f.attrs["measurement_quantity"] = eval_data.measurement_quantity
            f.attrs["measurement_names"] = eval_data.measurement_names

            results_group = f.create_group("results")

            for idx, single_res in enumerate(eval_data.results):
                res_subgroup = results_group.create_group(f"result_{idx}")

                res_subgroup.attrs["measurement"] = single_res.measurement or ""
                write_quantity(res_subgroup, "d", single_res.d)
                write_quantity(res_subgroup, "shift", single_res.shift)
                write_quantity(res_subgroup, "freq_axis", single_res.freq_axis)

                reg_grp = res_subgroup.create_group("regression_params")
                for k, v in single_res.regression_params.items():
                    if isinstance(v, Q_):
                        write_quantity(reg_grp, k, v)
                    elif isinstance(v, (int, float, str, bool)):
                        reg_grp.attrs[k] = v

                opt_grp = res_subgroup.create_group("optimization_info")
                for k, v in single_res.optimization_info.items():
                    if isinstance(v, Q_):
                        write_quantity(opt_grp, k, v)
                    elif isinstance(v, (int, float, str, bool)):
                        opt_grp.attrs[k] = v

                ds_grp = res_subgroup.create_group("datasets")
                for ds_name, q_dataset in single_res.datasets.items():
                    qd_group = ds_grp.create_group(ds_name)
                    quantity_dataset_to_hdf5group(q_dataset, qd_group)

        return fileName

    def process(self, eval_result):
        if not self.enabled:
            logging.info("Data storage is disabled, not saving results.")
            return

        filename = self._saveHDF5(eval_result)

        logging.info("Saved result as {}".format(filename))
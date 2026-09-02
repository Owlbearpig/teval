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

import numpy as np

from common.components import ComponentBase
from common.units import Q_

class DataSetDict(dict):
    def __init__(self, dataset_dict=None):
        super(DataSetDict, self).__init__()

        if dataset_dict is None:
            self[""] = DataSet()
        else:
            self.update(dataset_dict)

    def checkConsistency(self):
        for v in self.values():
            v.checkConsistency()

class DataSet:
    def __init__(self, data=None, uncert=None, axes=None, axes_labels=None, data_label=""):
        super().__init__()

        self.axes = [] if axes is None else axes
        self.data = Q_(np.array(0.0)) if data is None else data
        self.uncert = np.zeros_like(data) if uncert is None else uncert

        self.axes_labels = [] if axes_labels is None else axes_labels
        self.data_label = data_label

    @property
    def data_is_consistent(self):
        data_consistent = len(self.axes) == self.data.ndim and \
                          all([len(ax) == shape for (ax, shape) in zip(self.axes, self.data.shape)])
        uncert_consistent = len(self.axes) == self.uncert.ndim and \
                          all([len(ax) == shape for (ax, shape) in zip(self.axes, self.uncert.shape)])
        return data_consistent & uncert_consistent

    def checkConsistency(self):
        if not self.data_is_consistent:
            raise Exception("Data/uncertainty is inconsistent! "
                            "Number of axes: %d/%d, data dimension: %d/%d, "
                            "axes lengths: %s, data shape: %s/%s" %
                            (len(self.axes), len(self.uncert), self.data.ndim, self.uncert.ndim,
                             [len(ax) for ax in self.axes], self.uncert.shape, self.data.shape))

    def __repr__(self):
        return 'DataSet(%s, %s, %s)' % (repr(self.data), repr(self.uncert), repr(self.axes))

    def __str__(self):
        return 'DataSet with:\n    data: %s,\n    uncertainty: %s\n and axes:\n    %s' % \
                (repr(self.data).replace('\n', '\n    '),
                 repr(self.uncert).replace('\n', '\n    '),
                 repr(self.axes).replace('\n', '\n    '))

if __name__ == '__main__':

    freq_axis = np.ones(4001)

    test_dict = {
        "delta_n": np.ones((4001, 3)),
        "delta_alpha": np.ones(4001),
        "n0": np.ones(4001)*3.1,
        "n": np.ones(4001)*3.1415,
        "k": np.ones(4001)*3.1415,
        "alpha": np.ones(4001)*3.1415,
        "t_mod": np.ones(4001)*3.1415,
        "sam_mod": np.ones(4001)*3.1415,
    }
    ds1 = DataSet(axes=[freq_axis], data=test_dict["delta_n"][:, 1], uncert=test_dict["delta_n"][:, 2],
                  data_label="Simple n", axes_labels=["Frequency"])
    print(test_dict["delta_n"].shape)
    result_dict = {"1": ds1, "2": ds1}
    print(ds1)
    # print(result_dict["n0"])

    from common.traits import QuantityDict, DataSetDictClass

    class Test(ComponentBase):
        quant_dict = QuantityDict()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quant_dict = DataSetDictClass(result_dict)

    Test()
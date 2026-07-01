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
from common.units import Q_

class QuantityDict(dict):
    def __init__(self, data_dict=None, axes=None):
        super(QuantityDict, self).__init__()
        if data_dict is None:
            self[None] = DataSet()
        if axes is None:
            axes = []

        for k, v in data_dict.items():
            self[k] = DataSet(data=v, axes=axes)

    def checkConsistency(self):
        for k, v in self.items():
            v.checkConsistency()

class DataSet:
    def __init__(self, data=None, axes=None):
        super().__init__()

        if data is None:
            data = Q_(np.array(0.0))
        if axes is None:
            axes = []
        self.data = data
        self.axes = axes

    @property
    def isConsistent(self):
        return len(self.axes) == self.data.ndim and \
               all([len(ax) == shape
                    for (ax, shape) in zip(self.axes, self.data.shape)])

    def checkConsistency(self):
        if not self.isConsistent:
            raise Exception("DataSet is inconsistent! "
                            "Number of axes: %d, data dimension: %d, "
                            "axes lengths: %s, data shape: %s" %
                            (len(self.axes), self.data.ndim,
                             [len(ax) for ax in self.axes],
                             self.data.shape))

    def __repr__(self):
        return 'DataSet(%s, %s)' % (repr(self.data), repr(self.axes))

    def __str__(self):
        return 'DataSet with:\n    %s\n  and axes:\n    %s' % \
                (repr(self.data).replace('\n', '\n    '),
                 repr(self.axes).replace('\n', '\n    '))

freq_axis = np.ones(4001)

test_dict = {
    "delta_n": np.ones(4001),
    "delta_alpha": np.ones(4001),
    "n0": np.ones(4001)*3.1,
    "n": np.ones(4001)*3.1415,
    "k": np.ones(4001)*3.1415,
    "alpha": np.ones(4001)*3.1415,
    "t_mod": np.ones(4001)*3.1415,
    "sam_mod": np.ones(4001)*3.1415,
}

result_dict = QuantityDict(test_dict, axes=[freq_axis])

print(result_dict["n0"])

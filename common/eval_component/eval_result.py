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

class QuantitySet:



    def __init__(self, opt_res=None, data=None, axes=None):
        super().__init__()
        if opt_res is None:
            return

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

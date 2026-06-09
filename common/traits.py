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

import traitlets
from traitlets import TraitError, Undefined, TraitType, List as TList, Float, Integer

if float(traitlets.__version__[0]) <= 4:
    from traitlets import class_of
else:
    from traitlets.utils.descriptions import class_of
from .units import ureg, Q_
import pathlib


def instance_init(self, obj):
    with obj.cross_validation_lock:
        if self.default_value is not Undefined:
            v = self._validate(obj, self.default_value)
            if self.name is not None:
                self.set(obj, v)


TraitType = type(TraitType.__name__, (TraitType,), {"instance_init": instance_init})


class Path(TraitType):

    default_value = pathlib.Path()
    info_text = 'a path'

    def __init__(self, default_value=Undefined,
                 allow_none=None, **kwargs):
        self.is_file = kwargs.pop('is_file', True)
        self.is_dir = kwargs.pop('is_dir', True)
        self.must_exist = kwargs.pop('must_exist', True)
        super().__init__(default_value=default_value, allow_none=allow_none,
                         **kwargs)

    def validate(self, obj, value):
        if not isinstance(value, pathlib.Path):
            raise TraitError("'%s' is not a Path object!" % repr(value))

        if self.must_exist:
            if not value.exists():
                raise TraitError("The path '%s' does not exist" % value)
            if self.is_file and not self.is_dir and not value.is_file():
                raise TraitError("The path '%s' is not a file" % value)
            if not self.is_file and self.is_dir and not value.is_dir():
                raise TraitError("The path '%s' is not a directory" % value)
        return value


class Quantity(TraitType):

    default_value = Q_(0)
    info_text = 'a quantity'

    def __init__(self, default_value=Undefined,
                 allow_none=None, **kwargs):
        self.dimensionality = kwargs.pop('dimensionality', None)
        self.min = kwargs.pop('min', None)
        self.max = kwargs.pop('max', None)
        super().__init__(default_value=default_value, allow_none=allow_none,
                         **kwargs)

    def validate(self, obj, value):
        if not isinstance(value, ureg.Quantity):
            self.error(obj, value)
        if (self.dimensionality is not None and
                self.dimensionality != value.dimensionality):
            raise TraitError("The dimensionality of the '%s' trait of %s instance should "
                             "be %s, but a value with dimensionality %s was "
                             "specified" % (self.name, class_of(obj),
                                            self.dimensionality, value.dimensionality))

        if ((self.max is not None and (value.to(self.max.units) > self.max)) or
                (self.min is not None and (value.to(self.min.units) < self.min))):
            raise TraitError("The value of the '%s' trait of %s instance should "
                             "be between %s and %s, but a value of %s was "
                             "specified" % (self.name, class_of(obj),
                                            self.min, self.max, value))
        return value

class ValueRange(TraitType):
    info_text = 'a value range'
    default_value = [0.0, 1.0]

    def __init__(self, default_value=Undefined, allow_none=None, **kwargs):
        self.dimensionality = kwargs.pop('dimensionality', None)
        self.min = kwargs.pop('min', None)
        self.max = kwargs.pop('max', None)

        super().__init__(default_value=default_value, allow_none=allow_none,
                         **kwargs)

    def __getitem__(self, item):
        if isinstance(item, int):
            val = self.default_value[item]
            return val

    def validate(self, obj, value):
        if not len(value) == 2:
            self.error(obj, value)
        if not type(value[0]) == type(value[1]):
            self.error(obj, value)
        if isinstance(value[0], Q_) and value[0].units != value[1].units:
            self.error(obj, value)
        if not value[0] <= value[1]:
            self.error(obj, value)


        return value

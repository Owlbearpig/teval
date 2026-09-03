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
import typing as t

import traitlets
from common.eval_component.quantity_set import DataSetDict as DataSetDictClass
from traitlets import TraitError, Undefined, TraitType, List, Float, Integer, Instance, HasTraits

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


class TraitTypePatched(TraitType):
    def instance_init(self, obj: t.Any) -> None:
        with obj.cross_validation_lock:
            if self.default_value is not Undefined:
                v = self._validate(obj, self.default_value)
                if self.name is not None:
                    self.set(obj, v)

class QuantityDict(TraitTypePatched):

    default_value = DataSetDictClass()
    info_text = "Dict collection of datasets"

    def validate(self, obj, value):
        if isinstance(value, DataSetDictClass):
            value.checkConsistency()
            return value
        self.error(obj, value)

class TList(TraitTypePatched):
    default_value = []
    info_text = 'a list'

    def __init__(self, default_value=Undefined,
                 allow_none=None, **kwargs):
        super().__init__(default_value=default_value,
                         allow_none=allow_none,
                         **kwargs)
        self.selected_element = None



class Path(TraitTypePatched):

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

class MultiPathClass(HasTraits):
    selected_paths = List()

    def __init__(self, root_path=None, selected_paths=None, shown_filenames=None, **kwargs):
        super().__init__(**kwargs)
        self.root_path = root_path if root_path else pathlib.Path()
        self.selected_paths = selected_paths if selected_paths else []
        self.shown_filenames = shown_filenames if shown_filenames else []

    def exists(self, obj, value):
        if not isinstance(value.root_path, pathlib.Path):
            raise TraitError("'%s' is not a Path object!" % repr(value.root_path))
        if not value.root_path.exists():
            raise TraitError("The path '%s' does not exist" % value.root_path)

        for val in value.selected_paths:
            if not isinstance(val, pathlib.Path):
                raise TraitError("'%s' is not a Path object!" % repr(val))
            if not val.exists():
                raise TraitError("The path '%s' does not exist" % val)

class MultiPathSelection(TraitTypePatched):
    default_value = MultiPathClass()
    info_text = 'a multi path'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(self, obj, value):
        if isinstance(value, MultiPathClass):
            value.exists(obj, value)
            return value
        self.error(obj, value)

class Quantity(TraitTypePatched):

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


class QuantityList(list):
    @property
    def magnitude(self):
        return [
            v.magnitude if hasattr(v, "magnitude") else v
            for v in self
        ]

class ValueRange(TraitTypePatched):
    info_text = 'a value range'
    default_value = QuantityList([Q_(0, ""), Q_(0, "")])

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
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            self.error(obj, value)

        v0, v1 = value[0], value[1]

        if type(v0) != type(v1):
            self.error(obj, value)
        if hasattr(v0, "units") and hasattr(v1, "units") and v0.units != v1.units:
            self.error(obj, value)

        return QuantityList(value)

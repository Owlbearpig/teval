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

import pint
from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry()
Q_ = ureg.Quantity
set_application_registry(ureg)

if pint.__version__ == '0.7.2':

    UnitsContainer = pint.unit.UnitsContainer

    def __unbugged_format__(self, spec):
        spec = spec or self.default_format
        # special cases
        if 'Lx' in spec: # the LaTeX siunitx code
          opts = ''
          ustr = pint.unit.siunitx_format_unit(self)
          ret = r'\si[%s]{%s}'%( opts, ustr )
          return ret


        if '~' in spec:
            if not self._units:
                return ''
            units = UnitsContainer(dict((self._REGISTRY._get_symbol(key),
                                         value)
                                   for key, value in self._units.items()))
            spec = spec.replace('~', '')
        else:
            units = self._units

        return '%s' % (format(units, spec))

    pint.unit._Unit.__format__ = __unbugged_format__

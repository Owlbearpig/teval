import re
import numpy as np
from datetime import datetime
from pathlib import Path

default_time_format = "%Y-%m-%dT%H-%M-%S.%f"

meas_id_func = lambda meas_datetime: int((meas_datetime - datetime.min).total_seconds() * 1e6)
timestamp2id = lambda timestamp_str: meas_id_func(datetime.strptime(timestamp_str, default_time_format))

_number_regex = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_time_directive_regex = {
    "Y": r"\d{4}", "y": r"\d{2}", "m": r"\d{1,2}", "d": r"\d{1,2}", "H": r"\d{1,2}", "I": r"\d{1,2}",
    "M": r"\d{1,2}", "S": r"\d{1,2}", "f": r"\d{1,6}", "j": r"\d{1,3}", "p": r"[AaPp][Mm]",
    "b": r"[A-Za-z]+", "B": r"[A-Za-z]+", "a": r"[A-Za-z]+", "A": r"[A-Za-z]+", "z": r"[-+]\d{2}:?\d{2}|Z",
    "%": "%",
}
_placeholder = re.compile(r"\{(\w+)(?::(num|int|text))?}|\*")


def time_format_regex(time_format):
    # the time field must not be a lazy ".+?": it would end at the first separator that also occurs in the timestamp
    parts, pos = [], 0
    for match in re.finditer(r"%(.)", time_format):
        parts.append(re.escape(time_format[pos:match.start()]))
        parts.append(f"(?:{_time_directive_regex.get(match.group(1), '.+?')})")
        pos = match.end()
    parts.append(re.escape(time_format[pos:]))
    return "".join(parts)


class FileNameFormat:
    """
    How the measurement attributes are read from the file name (stem, without the extension).

    Empty pattern: the default format, the timestamp (default time format) at the start of the name and the
    positions from the "<value> mm" parts, e.g. "2025-05-27T23-29-21.791821-20avg-sam-X_25.000 mm-Y_-1.000 mm".

    Template: the whole file name has to match; "{field}" is a field, "*" anything. Fields:
    - time: measurement time, in the time format (without one, the modification time of the file)
    - x, y: position in mm (0 if missing)
    - name: measurement name, for the sample grouping by name
    - any other name: an extra attribute (number if it is one), "{field:num}" / "{field:int}" only match numbers
    E.g. "{time}-{avg:int}avg-*X_{x} mm-Y_{y} mm" or "{name}_{x}_{y}".

    A pattern starting with "re:" is a regular expression with the same named groups, e.g.
    "re:(?P<time>\\S+)_pos(?P<x>[-\\d.]+),(?P<y>[-\\d.]+)" (searched, doesn't have to match all of the name).
    """
    numeric_fields = ("x", "y")

    def __init__(self, pattern="", time_format=default_time_format):
        self.pattern = pattern.strip()
        self.time_format = time_format or default_time_format
        self.regex, self.fields, self.full_match = None, (), True
        if not self.pattern:
            return
        if self.pattern.startswith("re:"):
            self.regex = re.compile(self.pattern[3:])
            self.full_match = False
        else:
            self.regex = re.compile(self._template_regex(self.pattern))
        self.fields = tuple(self.regex.groupindex)

    def _template_regex(self, template):
        parts, pos, seen = [], 0, set()
        for match in _placeholder.finditer(template):
            parts.append(re.escape(template[pos:match.start()]))
            pos = match.end()
            if match.group(0) == "*":
                parts.append(".*?")
                continue
            field, kind = match.group(1), match.group(2)
            if field in seen:
                raise ValueError(f"Field {{{field}}} occurs twice in the file name format")
            seen.add(field)
            if field == "time":
                field_regex = time_format_regex(self.time_format)
            elif kind == "int":
                field_regex = r"[-+]?\d+"
            elif kind == "num" or field in self.numeric_fields:
                field_regex = _number_regex
            else:
                field_regex = ".+?"
            parts.append(f"(?P<{field}>{field_regex})")
        parts.append(re.escape(template[pos:]))
        return "".join(parts)

    @property
    def is_default(self):
        return self.regex is None

    def parse(self, filepath):
        # {field: value} of the file name, raises ValueError if it doesn't match
        stem = Path(filepath).stem
        match = self.regex.fullmatch(stem) if self.full_match else self.regex.search(stem)
        if match is None:
            raise ValueError(f"File name doesn't match the format \"{self.pattern}\"")
        values = {}
        for field, val in match.groupdict().items():
            if val is None:
                continue
            if field == "time":
                values[field] = datetime.strptime(val, self.time_format)
            elif field in self.numeric_fields:
                values[field] = float(val)
            elif field == "name":
                values[field] = val
            else:
                try:
                    values[field] = int(val) if re.fullmatch(r"[-+]?\d+", val) else float(val)
                except ValueError:
                    values[field] = val
        return values


default_file_name_format = FileNameFormat()


class Measurement:
    filepath = Path()
    meas_time = None
    sample_name = None
    position = (None, None)
    identifier = None

    def __init__(self, filepath=None, file_name_format=None):
        self.filepath = filepath
        self.file_name_format = file_name_format or default_file_name_format
        # measurement name from the {name} field, None: derived from the file name (MeasurementSelection)
        self.name = None
        # extra fields of the file name format
        self.attributes = {}
        self.time_from_file = False

        self._data_fd, self._data_td = None, None

        self._set_metadata()

    def __repr__(self):
        return str(self.filepath)

    def _extract_position(self):
        fp_stem = str(self.filepath.stem)

        matches = re.findall(r"(-?\d+\.\d+|-?\d+) mm", fp_stem)
        positions = [0.000 if np.isclose(float(val), 0) else float(val) for val in matches]
        l_diff = 2 - len(positions)
        if l_diff > 0:
            positions.extend(l_diff * [0.0])
        positions = tuple(positions)

        return positions

    def _set_metadata(self):
        if self.file_name_format.is_default:
            date_string = str(self.filepath.stem)[:26]
            self.meas_time = datetime.strptime(date_string, default_time_format)
            self.position = self._extract_position()
        else:
            values = self.file_name_format.parse(self.filepath)
            if "time" in values:
                self.meas_time = values.pop("time")
            else:
                self.meas_time = datetime.fromtimestamp(self.filepath.stat().st_mtime)
                self.time_from_file = True
            self.position = tuple(0.0 if np.isclose(val, 0) else val
                                  for val in (values.pop("x", 0.0), values.pop("y", 0.0)))
            self.name = values.pop("name", None)
            self.attributes = values

        dir_1above, dir_2above = self.filepath.parents[0], self.filepath.parents[1]
        if ("sam" in dir_1above.stem.lower()) or ("ref" in dir_1above.stem.lower()):
            self.sample_name = dir_2above.stem
        else:
            self.sample_name = dir_1above.stem

        self.identifier = meas_id_func(self.meas_time)

    def get_data(self):
        if self._data_td is None:
            self._data_td = np.loadtxt(self.filepath)

        return self._data_td

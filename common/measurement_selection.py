from common.components import ComponentBase
from common.traits import MultiPathSelection, ValueRange, MultiPathClass
from common.units import Q_
from traitlets import Enum as TEnum, Unicode, Bool
from enum import Enum

class SelectionCriterionEnum(Enum):
    file_selection = "File selection"
    selected_timestamp = "Timestamp"
    selected_point = "Point"
    string_search = "String"

class MeasurementSelection(ComponentBase):

    selection_criterion = TEnum(SelectionCriterionEnum,
                                SelectionCriterionEnum.file_selection).tag(name="Select measurement by", priority=-2)
    sel_point = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(name="Selected point (x, y)")
    sel_timestamp = Unicode("").tag(name="Selected timestamp")
    string_match = Unicode("").tag(name="Filter string")
    allow_averaging = Bool(False).tag(name="Allow averaging")

    references = MultiPathSelection().tag(fullwidth = False, group="Reference selection", combine=True)
    samples = MultiPathSelection().tag(fullwidth = False, group="Sample selection", combine=True)


    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)

        self.dataset = dataset

        self.references = MultiPathClass(root_path=self.dataset.data_path)
        self.samples = MultiPathClass(root_path=self.dataset.data_path)

        self.dataset.observe(self.update, "data_path")

    def update(self, change):
        new_path = change["new"]
        self.references = MultiPathClass(root_path=new_path)
        self.samples = MultiPathClass(root_path=new_path)

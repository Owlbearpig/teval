from common.components import ComponentBase
from common.traits import QuantityDict
from common.eval_component.quantity_set import QuantityDict as QuantityDictClass
from traitlets import Bool, Float, Int, Unicode
from common.traits import Quantity, Q_
import numpy as np

class EvalResult(ComponentBase):

    quantity_dict = QuantityDict()

    d = Quantity(Q_(0, "µm"), read_only=True)
    q_val = Float(0.0, read_only=True)
    gof = Float(0.0, read_only=True)
    shift = Quantity(Q_(0.0, "fs"), read_only=True)
    timestamp = Unicode("", read_only=True)
    converged = Bool(False, read_only=True)

    def __init__(self, opt_res_dict=None):
        super().__init__()
        if opt_res_dict is None:
            return

        self.process_dict(opt_res_dict)

    def process_dict(self, opt_res_dict):
        for k, v in opt_res_dict.items():
            if isinstance(v, str):
                self.set_trait(k, v)
            elif (k in self.attributes) and isinstance(self.attributes[k], Quantity):
                unit = self.attributes[k].default_value.units
                self.set_trait(k, v * unit)
            elif isinstance(v, (int, float)):
                self.set_trait(k, v)

        arr_dict = {k: v for k, v in opt_res_dict.items() if
                    isinstance(v, np.ndarray) and (v.ndim == 2 and v.shape[1] == 2)}
        self.quantity_dict = QuantityDictClass(arr_dict)

from common.components import ComponentBase
from common.traits import QuantitySet
from traitlets import Bool, Float, Int
from common.traits import Quantity, Q_

class EvalResult(ComponentBase):

    quantity_set = QuantitySet()

    d = Quantity(Q_(0, "µm"), read_only=True)
    q_val = Float(0.0, read_only=True)
    gof = Float(0.0, read_only=True)
    shift = Quantity(Q_(0.0, "fs"), read_only=True)

    def __init__(self, opt_res_dict=None):
        if opt_res_dict is None:
            return

        self.process(opt_res_dict)

    def process(self, opt_res_dict):
        for k, v in opt_res_dict.items():
            if isinstance(v, (float, int, str)):
                self.set_trait(k, v)
            else:
                # vectors: collect and pass to QuantitySet
                pass

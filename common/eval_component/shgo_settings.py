from common.components import ComponentBase
from enum import Enum
from traitlets import Float, Integer, Bool, Instance, Enum as TEnum

class MinimizerMethod(Enum):
    NelderMead = "Nelder-Mead"

class MinimizerOptions(ComponentBase):
    minimizer_opt_grp = "options"
    method = TEnum(MinimizerMethod, default_value=MinimizerMethod.NelderMead, group="method")
    maxfev = Integer(200, group=minimizer_opt_grp)
    maxev = Integer(200, group=minimizer_opt_grp)
    maxiter = Integer(200, group=minimizer_opt_grp)
    tol = Float(1e-13, group=minimizer_opt_grp)
    fatol = Float(1e-13, group=minimizer_opt_grp)
    xatol = Float(1e-13, group=minimizer_opt_grp)

class SHGOOptions(ComponentBase):
    n = Integer(2)
    iters = Integer(100)

    shgo_options_grp = "shgo_options"
    maxfev = Integer(1500, group=shgo_options_grp)
    f_tol = Float(1e-12, group=shgo_options_grp)
    maxiter = Integer(4000, group=shgo_options_grp)
    xtol = Float(1e-12, group=shgo_options_grp)
    maxev = Integer(4000, group=shgo_options_grp)
    minimize_every_iter = Bool(False, group=shgo_options_grp)
    disp = Bool(False, group=shgo_options_grp)

    minimizer_kwargs = Instance(MinimizerOptions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.minimizer_kwargs = MinimizerOptions()


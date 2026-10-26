from common.components import ComponentBase
from enum import Enum
from traitlets import Float, Integer, Bool, Instance, Enum as TEnum

class MinimizerMethod(Enum):
    SLSQP = "SLSQP"
    NelderMead = "Nelder-Mead"
    Powell = "Powell"
    LBFGSB = "L-BFGS-B"

method_option_names = {
    MinimizerMethod.SLSQP: {"maxiter": "maxiter", "ftol": "ftol"},
    MinimizerMethod.NelderMead: {"maxiter": "maxiter", "maxfev": "maxfev", "ftol": "fatol", "xtol": "xatol"},
    MinimizerMethod.Powell: {"maxiter": "maxiter", "maxfev": "maxfev", "ftol": "ftol", "xtol": "xtol"},
    MinimizerMethod.LBFGSB: {"maxiter": "maxiter", "maxfev": "maxfun", "ftol": "ftol"},
}

class MinimizerOptions(ComponentBase):
    minimizer_opt_grp = "Local minimizer"
    method = TEnum(MinimizerMethod, default_value=MinimizerMethod.SLSQP,
                   help="scipy.optimize.minimize method of the local searches").tag(
        name="Method", group=minimizer_opt_grp, priority=0)
    tol = Float(0.0, min=0.0, help="tol argument of scipy.optimize.minimize, 0: not set").tag(
        name="Tolerance (tol, 0: not set)", group=minimizer_opt_grp, priority=1)
    maxiter = Integer(0, min=0, help="Maximum iterations of a local search, 0: default of the method").tag(
        name="Max. iterations (0: default)", group=minimizer_opt_grp, priority=2)
    maxfev = Integer(0, min=0, help="Maximum function evaluations of a local search (maxfev, maxfun for "
                                    "L-BFGS-B; not used by SLSQP), 0: default of the method").tag(
        name="Max. function evaluations (0: default)", group=minimizer_opt_grp, priority=3)
    ftol = Float(0.0, min=0.0, help="Function value tolerance (ftol, fatol for Nelder-Mead), 0: default of the "
                                    "method").tag(name="Function tolerance (0: default)", group=minimizer_opt_grp,
                                                  priority=4)
    xtol = Float(0.0, min=0.0, help="Parameter tolerance (xatol for Nelder-Mead, xtol for Powell; not used by "
                                    "SLSQP and L-BFGS-B), 0: default of the method").tag(
        name="Parameter tolerance (0: default)", group=minimizer_opt_grp, priority=5)

    def get_minimizer_kwargs(self):
        options = {scipy_name: getattr(self, name) for name, scipy_name in method_option_names[self.method].items()
                   if getattr(self, name) > 0}
        kwargs = {"method": self.method.value, "options": options}
        if self.tol > 0:
            kwargs["tol"] = self.tol
        return kwargs

class SHGOOptions(ComponentBase):
    shgo_options_grp = "Global search (shgo)"
    stopping_names = ("maxiter", "maxfev", "maxev")

    n = Integer(2, min=1, help="Sampling points per iteration").tag(name="Sampling points (n)",
                                                                     group=shgo_options_grp, priority=0)
    iters = Integer(100, min=1, help="Iterations of the sampling, only used if max. iterations, max. function "
                                     "evaluations and max. sampling evaluations are all 0").tag(
        name="Iterations (iters)", group=shgo_options_grp, priority=1)
    maxiter = Integer(4000, min=0, help="Stopping criterion, 0: not set. Any stopping criterion replaces "
                                        "iters").tag(name="Max. iterations (0: not set)", group=shgo_options_grp,
                                                     priority=2)
    maxfev = Integer(1500, min=0, help="Stopping criterion: function evaluations in the feasible domain, 0: not "
                                       "set. Any stopping criterion replaces iters").tag(
        name="Max. function evaluations (0: not set)", group=shgo_options_grp, priority=3)
    maxev = Integer(4000, min=0, help="Stopping criterion: sampling evaluations, 0: not set. Any stopping criterion "
                                      "replaces iters").tag(name="Max. sampling evaluations (0: not set)",
                                                            group=shgo_options_grp, priority=4)
    minimize_every_iter = Bool(False, help="Run the local searches after every iteration, otherwise only at the "
                                           "end").tag(name="Minimize every iteration", group=shgo_options_grp,
                                                      priority=5)
    disp = Bool(False, help="Log the progress of the global search").tag(name="Log progress",
                                                                         group=shgo_options_grp, priority=6)

    minimizer_kwargs = Instance(MinimizerOptions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.minimizer_kwargs = MinimizerOptions()

    def get_shgo_options(self):
        options = {"minimize_every_iter": self.minimize_every_iter, "disp": self.disp}
        options.update({name: getattr(self, name) for name in self.stopping_names if getattr(self, name) > 0})
        return options

    def get_minimizer_kwargs(self):
        return self.minimizer_kwargs.get_minimizer_kwargs()

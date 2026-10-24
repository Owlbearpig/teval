from dataclasses import dataclass, field

import numpy as np

from common.eval_component.eval_result import SingleResultData


class OptimizationCancelled(Exception):
    """Raised inside the evaluation thread when the optimization is canceled."""


@dataclass
class ScanTarget:
    # one thickness / shift scan with its own data, e.g. a measurement of a selection or an image pixel
    key: object
    opt_config: dict
    d_axis: np.ndarray
    d_step: float
    results: list = field(default_factory=list)
    done_keys: set = field(default_factory=set)
    best: SingleResultData | None = None

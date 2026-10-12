import numpy as np
from enum import Enum, member


def abs_cost_fun(y_meas, y_mod):
    abs_diff = (np.abs(y_meas) - np.abs(y_mod)) ** 2

    return abs_diff


def phi_cost_fun(y_meas, y_mod):
    phi_diff = np.angle(y_meas / y_mod) ** 2

    return phi_diff


def combined_cost_fun(y_meas, y_mod):
    return abs_cost_fun(y_meas, y_mod) + phi_cost_fun(y_meas, y_mod)

class CostFunctions(Enum):
    abs_cost = member(abs_cost_fun)
    phi_cost = member(phi_cost_fun)
    combined_cost = member(combined_cost_fun)
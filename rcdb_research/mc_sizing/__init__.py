from ctypes import cdll, c_double, c_bool, c_int
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
lib = cdll.LoadLibrary(os.path.join(dir_path, 'lib.so'))
fun = lib.compute_bet_old
fun.argtypes = (c_double, c_double, c_double, c_double, c_bool, c_double, c_double, c_double, c_int, c_int)
fun.restype = c_double


def compute_bet_old(
        avg_win: float,
        avg_loss: float,
        minimum_wealth: float,
        max_drawdown_risk: float,
        compounded: bool,
        predicted_probability: float,
        size_upper_bound: float,
        xtol: float,
        n_curves: int,
        n_steps: int
):
    return fun(
        avg_win,
        avg_loss,
        minimum_wealth,
        max_drawdown_risk,
        compounded,
        predicted_probability,
        size_upper_bound,
        xtol,
        n_curves,
        n_steps
    )


fun_new = lib.compute_bet
fun_new.argtypes = (c_double, c_double, c_double, c_double, c_bool, c_double, c_double, c_double, c_int, c_int)
fun_new.restype = c_double


def compute_bet(
        avg_win: float,
        avg_loss: float,
        minimum_wealth: float,
        max_drawdown_risk: float,
        compounded: bool,
        predicted_probability: float,
        size_upper_bound: float,
        xtol: float,
        n_curves: int,
        n_steps: int
):
    return fun_new(
        avg_win,
        avg_loss,
        minimum_wealth,
        max_drawdown_risk,
        compounded,
        predicted_probability,
        size_upper_bound,
        xtol,
        n_curves,
        n_steps
    )


__all__ = ['compute_bet_old', 'compute_bet']

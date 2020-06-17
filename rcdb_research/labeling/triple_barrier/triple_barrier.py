# flake8: noqa


from ctypes import *
import numpy as np
import os
from typing import Union, Optional
import pandas as pd


dir_path = os.path.dirname(os.path.abspath(__file__))
lib = cdll.LoadLibrary(os.path.join(dir_path, 'lib.so'))
fun = lib.triple_barrier
fun.argtypes = (
	POINTER(c_double),
	c_int, 
	POINTER(c_int), 
	POINTER(c_double),
	POINTER(c_double), 
	POINTER(c_double), 
	POINTER(c_bool), 
	POINTER(c_int)
)
fun.restype = None


def triple_barrier_barebones(
	close: np.ndarray,
	vertical_barrier: np.ndarray,
	unit_width: np.ndarray,
	pt_mul: np.ndarray,
	sl_mul: np.ndarray,
	event: np.ndarray
):
	assert close.dtype == c_double
	assert vertical_barrier.dtype == c_int
	assert unit_width.dtype == c_double
	assert pt_mul is None or pt_mul.dtype == c_double
	assert sl_mul is None or sl_mul.dtype == c_double
	assert event.dtype == c_bool

	n_bars = close.shape[0]
	result = np.empty(np.sum(event), dtype=c_int)

	lib.triple_barrier(
		close.ctypes.data_as(POINTER(c_double)),
		close.shape[0],
		vertical_barrier.ctypes.data_as(POINTER(c_int)),
		unit_width.ctypes.data_as(POINTER(c_double)),
		pt_mul.ctypes.data_as(POINTER(c_double)) if pt_mul is not None else None,
		sl_mul.ctypes.data_as(POINTER(c_double)) if sl_mul is not None else None,
		event.ctypes.data_as(POINTER(c_bool)),
		result.ctypes.data_as(POINTER(c_int)),
	)

	return result


def toarray(a):
    if isinstance(a, np.ndarray):
        return a
    elif isinstance(a, pd.Series):
        return a.values
    else:
        return np.array(a)


def triple_barrier_singlethreaded(
    close: pd.Series,
    unit_width: Union[float, np.ndarray],
    pt_mul: Optional[Union[float, np.ndarray]] = None,
    sl_mul: Optional[Union[float, np.ndarray]] = None,
    vertical_barrier: Optional[Union[int, np.ndarray]] = None,
    event: Optional[np.ndarray] = None
):
    if vertical_barrier is None:
        vertical_barrier = np.full(close.size, close.index.values.max())
    if np.isscalar(vertical_barrier):
        vertical_barrier = np.hstack([close.index[vertical_barrier:], np.full(vertical_barrier, close.index.max())])
    if np.isscalar(pt_mul):
        pt_mul = np.full(close.size, pt_mul)
    if np.isscalar(sl_mul):
        sl_mul = np.full(close.size, sl_mul)
    if np.isscalar(unit_width):
        unit_width = np.full(close.size, unit_width)
    if event is None:
        event = np.full(close.size, True)
    
    vertical_barrier = np.searchsorted(close.index, vertical_barrier).astype(c_int)

    result = triple_barrier_barebones(
        close.values.astype(c_double),
        toarray(vertical_barrier).astype(c_int),
        toarray(unit_width).astype(c_double),
        toarray(pt_mul).astype(c_double) if pt_mul is not None else None,
        toarray(sl_mul).astype(c_double) if sl_mul is not None else None,
        toarray(event).astype(c_bool)
    )
    
    return pd.Series(name='t1', index=close.index[event], data=close.index[result])


def triple_barrier_barebones_threaded(
	close: np.ndarray,
	vertical_barrier: np.ndarray,
	unit_width: np.ndarray,
	pt_mul: np.ndarray,
	sl_mul: np.ndarray,
	event: np.ndarray,
	n_jobs
):
	assert close.dtype == c_double
	assert vertical_barrier.dtype == c_int
	assert unit_width.dtype == c_double
	assert pt_mul is None or pt_mul.dtype == c_double
	assert sl_mul is None or sl_mul.dtype == c_double
	assert event.dtype == c_bool

	result = np.empty(np.sum(event), dtype=c_int)

	lib.triple_barrier_threaded(
		close.ctypes.data_as(POINTER(c_double)),
		c_int(close.shape[0]),
		vertical_barrier.ctypes.data_as(POINTER(c_int)),
		unit_width.ctypes.data_as(POINTER(c_double)),
		pt_mul.ctypes.data_as(POINTER(c_double)) if pt_mul is not None else None,
		sl_mul.ctypes.data_as(POINTER(c_double)) if sl_mul is not None else None,
		event.ctypes.data_as(POINTER(c_bool)),
		c_int(n_jobs),
		result.ctypes.data_as(POINTER(c_int)),
	)

	return result


def triple_barrier(
    close: pd.Series,
    unit_width: Union[float, np.ndarray],
    pt_mul: Optional[Union[float, np.ndarray]] = None,
    sl_mul: Optional[Union[float, np.ndarray]] = None,
    vertical_barrier: Optional[Union[int, np.ndarray]] = None,
    event: Optional[np.ndarray] = None,
    n_jobs: int = 1
):
    if vertical_barrier is None:
        vertical_barrier = np.full(close.size, close.index.values.max())
    if np.isscalar(vertical_barrier):
        vertical_barrier = np.hstack([close.index[vertical_barrier:], np.full(vertical_barrier, close.index.max())])
    if np.isscalar(pt_mul):
        pt_mul = np.full(close.size, pt_mul)
    if np.isscalar(sl_mul):
        sl_mul = np.full(close.size, sl_mul)
    if np.isscalar(unit_width):
        unit_width = np.full(close.size, unit_width)
    if event is None:
        event = np.full(close.size, True)
    
    vertical_barrier = np.searchsorted(close.index, vertical_barrier).astype(c_int)

    result = triple_barrier_barebones_threaded(
        close.values.astype(c_double),
        toarray(vertical_barrier).astype(c_int),
        toarray(unit_width).astype(c_double),
        toarray(pt_mul).astype(c_double) if pt_mul is not None else None,
        toarray(sl_mul).astype(c_double) if sl_mul is not None else None,
        toarray(event).astype(c_bool),
        n_jobs
    )
    
    return pd.Series(name='t1', index=close.index[event], data=close.index[result])



from typing import List

import numpy as np

from .utils import mne_doc_helper
from ..utils import rolling_window, generate_calc_all, feature_registrator_factory

from mne_features import bivariate, univariate

PREFIX = "mne"
FEATURE_FUNCS = {}

register_feature_mne = feature_registrator_factory(FEATURE_FUNCS)


@register_feature_mne
@mne_doc_helper(univariate.compute_ptp_amp)
def ptp_amp(series: np.array, window: int) -> np.array:
    """
    Peak-to-peak (PTP) amplitude of the data

    :param series:
    :param window:
    :return:
    """
    return univariate.compute_ptp_amp(
        rolling_window(series, window)
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_hurst_exp)
def hurst(series: np.array, window: int) -> np.array:
    """
    Hurst exponent of the data

    :param series:
    :param window:
    :return:
    """
    return univariate.compute_hurst_exp(
        rolling_window(series, window)
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_decorr_time)
def decorr_time(series: np.array, window: int, sfreq: float = 256.) -> np.array:
    """
    Decorrelation time

    :param series:
    :param window:
    :param sfreq:
    :return:
    """
    return univariate.compute_decorr_time(
        sfreq,
        rolling_window(series, window)
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_pow_freq_bands)
def pow_freq_bands(
        series: np.array,
        window: int,
        sfreq: float = 256.,
        freq_bands: np.array = np.array([0.5, 4., 8., 13., 30., 100.]),
        normalize: bool = True,
        ratios: str = None,
        psd_method: str = 'welch',
        psd_params: dict = None) -> np.array:
    """
    Power Spectrum (computed by frequency bands)

    :param series:
    :param window:
    :param sfreq:
    :param freq_bands:
    :param normalize:
    :param ratios:
    :param psd_method:
    :param psd_params:
    :return:
    """
    data = rolling_window(series, window)
    res = univariate.compute_pow_freq_bands(
        sfreq,
        data,
        freq_bands=freq_bands,
        normalize=normalize,
        ratios=ratios,
        psd_method=psd_method,
        psd_params=psd_params
    )
    return res


@register_feature_mne
@mne_doc_helper(univariate.compute_hjorth_mobility_spect)
def hjorth_mobility_spect(
        series: np.array,
        window: int,
        sfreq: float = 256.,
        normalize: bool = False,
        psd_method: str = 'welch',
        psd_params: dict = None) -> np.array:
    """
    Hjorth mobility

    :param series:
    :param window:
    :param sfreq:
    :param normalize:
    :param psd_method:
    :param psd_params:
    :return:
    """
    return univariate.compute_hjorth_mobility_spect(
        sfreq,
        rolling_window(series, window),
        normalize=normalize,
        psd_method=psd_method,
        psd_params=psd_params
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_hjorth_complexity_spect)
def hjorth_complexity_spect(
        series: np.array,
        window: int,
        sfreq: float = 256.) -> np.array:
    """
    Hjorth complexity

    :param series:
    :param window:
    :param sfreq:
    :return:
    """
    return univariate.compute_hjorth_complexity_spect(
        sfreq,
        rolling_window(series, window)
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_hjorth_mobility)
def hjorth_mobility(series: np.array, window: int) -> np.array:
    """
    Hjorth mobility

    :param series:
    :param window:
    :return:
    """
    return univariate.compute_hjorth_mobility(
        rolling_window(series, window)
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_hjorth_complexity)
def hjorth_complexity(series: np.array, window: int) -> np.array:
    """
    Hjorth complexity

    :param series:
    :param window:
    :return:
    """
    return univariate.compute_hjorth_complexity(
        rolling_window(series, window)
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_zero_crossings)
def zero_cross(series: np.array, window: int) -> np.array:
    """
    Number of zero crossings

    :param series:
    :param window:
    :return:
    """
    return univariate.compute_zero_crossings(
        rolling_window(series, window)
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_line_length)
def line_length(series: np.array, window: int) -> np.array:
    """
    Line length

    :param series:
    :param window:
    :return:
    """
    return univariate.compute_line_length(
        rolling_window(series, window)
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_spect_slope)
def spect_slope(
        series: np.array,
        window: int,
        sfreq: float = 256.,
        fmin: float = 0.1,
        fmax: float = 50,
        with_intercept: bool = True,
        psd_method: str = 'welch',
        psd_params: dict = None) -> np.array:
    """
    Linear regression of the the log-log frequency-curve

    :param series:
    :param window:
    :param sfreq:
    :param fmin:
    :param fmax:
    :param with_intercept:
    :param psd_method:
    :param psd_params:
    :return:
    """
    res = univariate.compute_spect_slope(
        sfreq,
        rolling_window(series, window)[window - 1:],
        fmin=fmin,
        fmax=fmax,
        with_intercept=with_intercept,
        psd_method=psd_method,
        psd_params=psd_params
    )
    return np.hstack(
        (
            [np.nan for _ in range(window - 1)],
            res
        )
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_svd_fisher_info)
def svd_fisher_info(
        series: np.array,
        window: int,
        tau: int = 2,
        emb: int = 10) -> np.array:
    """
    SVD Fisher Information

    :param series:
    :param window:
    :param tau:
    :param emb:
    :return:
    """
    res = univariate.compute_svd_fisher_info(
        rolling_window(series, window)[window - 1:],
        tau=tau,
        emb=emb
    )
    return np.hstack(
        (
            [np.nan for _ in range(window - 1)],
            res
        )
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_energy_freq_bands)
def energy_freq_bands(
        series: np.array,
        window: int,
        sfreq: float = 256.,
        freq_bands: np.array = np.array([0.5, 4., 8., 13., 30., 100.]),
        deriv_filt: bool = True) -> np.array:
    """
    Band energy

    :param series:
    :param window:
    :param sfreq:
    :param freq_bands:
    :param deriv_filt:
    :return:
    """
    return univariate.compute_energy_freq_bands(
        sfreq,
        rolling_window(series, window),
        freq_bands=freq_bands,
        deriv_filt=deriv_filt
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_spect_edge_freq)
def spect_edge_freq(
        series: np.array,
        window: int,
        sfreq: float = 256.,
        ref_freq: float = None,
        edge: List[float] = None,
        psd_method: str = 'welch',
        psd_params: dict = None) -> np.array:
    """
    Spectal Edge Frequency

    :param series:
    :param window:
    :param sfreq:
    :param ref_freq:
    :param edge:
    :param psd_method:
    :param psd_params:
    :return:
    """
    return univariate.compute_spect_edge_freq(
        sfreq,
        rolling_window(series, window),
        ref_freq=ref_freq,
        edge=edge,
        psd_method=psd_method,
        psd_params=psd_params
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_wavelet_coef_energy)
def wavelet_coef_energy(series: np.array, window: int, wavelet_name: str = 'db4') -> np.array:
    """
    Energy of Wavelet decomposition coefficients

    :param series:
    :param window:
    :param wavelet_name:
    :return:
    """
    return univariate.compute_wavelet_coef_energy(
        rolling_window(series, window),
        wavelet_name=wavelet_name
    )


@register_feature_mne
@mne_doc_helper(univariate.compute_teager_kaiser_energy)
def teager_kaiser_energy(series: np.array, window: int, wavelet_name: str = 'db4') -> np.array:
    """
    Compute the Teager-Kaiser energy

    :param series:
    :param window:
    :param wavelet_name:
    :return:
    """
    return univariate.compute_teager_kaiser_energy(
        rolling_window(series, window),
        wavelet_name=wavelet_name
    )


# @register_feature_mne
@mne_doc_helper(bivariate.compute_max_cross_corr)
def max_cross_corr(
        series: np.array,
        window: int,
        sfreq: float = 256.,
        include_diag: bool = False) -> np.array:
    """
    Maximum linear cross-correlation

    :param series:
    :param window:
    :param sfreq:
    :param include_diag:
    :return:
    """
    return bivariate.compute_max_cross_corr(
        sfreq,
        rolling_window(series, window),
        include_diag=include_diag
    )


@register_feature_mne
@mne_doc_helper(bivariate.compute_phase_lock_val)
def phase_lock_val(
        series: np.array,
        window: int,
        include_diag: bool = False) -> np.array:
    """
    Phase Locking Value (PLV)

    :param series:
    :param window:
    :param include_diag:
    :return:
    """
    res = bivariate.compute_phase_lock_val(
        rolling_window(series, window)[window - 1:],
        include_diag=include_diag
    )
    cols = res.reshape(-1, res.size // (series.size - window + 1)).transpose()
    res = np.empty((len(cols), series.size)) * np.nan
    for i in range(len(cols)):
        res[i, window - 1:] = cols[i]
    return res


@register_feature_mne
@mne_doc_helper(bivariate.compute_nonlin_interdep)
def nonlin_interdep(
        series: np.array,
        window: int,
        tau: int = 2,
        emb: int = 10,
        nn: int = 5,
        include_diag: bool = False) -> np.array:
    """
    Measure of nonlinear interdependence

    :param series:
    :param window:
    :param tau:
    :param emb:
    :param nn:
    :param include_diag:
    :return:
    """
    res = bivariate.compute_nonlin_interdep(
        rolling_window(series, window)[window - 1:],
        tau=tau,
        emb=emb,
        nn=nn,
        include_diag=include_diag
    )
    return np.hstack(
        (
            [np.nan for _ in range(window - 1)],
            res
        )
    )


@register_feature_mne
@mne_doc_helper(bivariate.compute_time_corr)
def time_corr(
        series: np.array,
        window: int,
        with_eigenvalues: bool = True,
        include_diag: bool = False) -> np.array:
    """
    Correlation Coefficients

    :param series:
    :param window:
    :param with_eigenvalues:
    :param include_diag:
    :return:
    """
    res = bivariate.compute_time_corr(
        rolling_window(series, window)[window - 1:],
        with_eigenvalues=with_eigenvalues,
        include_diag=include_diag
    )
    return np.hstack(([np.nan for _ in range(window - 1)], res))


@register_feature_mne
@mne_doc_helper(bivariate.compute_spect_corr)
def spect_corr(
        series: np.array,
        window: int,
        sfreq: float = 256.,
        with_eigenvalues: bool = True,
        include_diag: bool = False,
        psd_method: str = 'welch',
        psd_params: dict = None) -> np.array:
    """
    Correlation Coefficients

    :param series:
    :param window:
    :param sfreq:
    :param with_eigenvalues:
    :param include_diag:
    :param psd_method:
    :param psd_params:
    :return:
    """
    res = bivariate.compute_spect_corr(
        sfreq,
        rolling_window(series, window)[window - 1:],
        with_eigenvalues=with_eigenvalues,
        include_diag=include_diag,
        psd_method=psd_method,
        psd_params=psd_params
    )
    return np.hstack(([np.nan for _ in range(window - 1)], res))


__all__ = ("FEATURE_FUNCS", "PREFIX", "calc_all", *FEATURE_FUNCS.keys())

calc_all = generate_calc_all(PREFIX, FEATURE_FUNCS)

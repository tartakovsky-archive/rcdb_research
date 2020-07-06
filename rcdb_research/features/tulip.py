import numpy as np
from tulipindicators import ti


def adosc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    short_period: int,
    long_period: int
) -> np.ndarray:
    """Accumulation/Distribution Oscillator

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.
    short_period : int
        Short period parameter
    long_period : int
        Long period parameter

    Returns
    -------
    np.ndarray
        adosc

    """
    return ti.adosc(high, low, close, volume, short_period, long_period)


def adx(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Movement Index

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        adx

    """
    return ti.adx(high, low, period)


def adxr(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Movement Rating

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        adxr

    """
    return ti.adxr(high, low, period)


def ao(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Awesome Oscillator

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        ao

    """
    return ti.ao(high, low)


def apo(series: np.ndarray, short_period: int, long_period: int) -> np.ndarray:
    """Absolute Price Oscillator

    Parameters
    ----------
    series : np.ndarray
        Input data.
    short_period : int
        Short period parameter
    long_period : int
        Long period parameter

    Returns
    -------
    np.ndarray
        apo

    """
    return ti.apo(series, short_period, long_period)


def aroonosc(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Aroon Oscillator

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        aroonosc

    """
    return ti.aroonosc(high, low, period)


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Commodity Channel Index

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        cci

    """
    return ti.cci(high, low, close, period)


def cmf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Chaikin Money Flow

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        cmf

    """
    return ti.cmf(high, low, close, volume, period)


def cmo(series: np.ndarray, period: int) -> np.ndarray:
    """Chande Momentum Oscillator

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        cmo

    """
    return ti.cmo(series, period)


def copp(series: np.ndarray, roc_shorter_period: int, roc_longer_period: int, wma_period: int) -> np.ndarray:
    """Coppock Curve

    Parameters
    ----------
    series : np.ndarray
        Input data.
    roc_shorter_period : int
        ROC shorter period
    roc_longer_period : int
        ROC longer period
    wma_period : int
        WMA period

    Returns
    -------
    np.ndarray
        copp

    """
    return ti.copp(series, roc_shorter_period, roc_longer_period, wma_period)


def cvi(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Chaikins Volatility

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        cvi

    """
    return ti.cvi(high, low, period)


def dpo(series: np.ndarray, period: int) -> np.ndarray:
    """Detrended Price Oscillator

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        dpo

    """
    return ti.dpo(series, period)


def dx(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Directional Movement Index

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        dx

    """
    return ti.dx(high, low, period)


def emv(high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Ease of Movement

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        emv

    """
    return ti.emv(high, low, volume)


def fisher(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Fisher Transform

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        fisher

    """
    return ti.fisher(high, low, period).fisher


def fisher_signal(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Fisher Transform

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        fisher

    """
    return ti.fisher(high, low, period).fisher_signal


def fosc(series: np.ndarray, period: int) -> np.ndarray:
    """Forecast Oscillator

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        fosc

    """
    return ti.fosc(series, period)


def linregslope(series: np.ndarray, period: int) -> np.ndarray:
    """Linear Regression Slope

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        linregslope

    """
    return ti.linregslope(series, period)


def macd(series: np.ndarray, short_period, long_period, signal_period) -> np.ndarray:
    """Moving Average Convergence/Divergence

    Parameters
    ----------
    series : np.ndarray
        Input data.
    short_period : int
        Short period
    long_period : int
        Long period
    signal_period : int
        Signal period

    Returns
    -------
    np.ndarray
        macd

    """
    return ti.macd(series, short_period, long_period, signal_period).macd


def macd_signal(series: np.ndarray, short_period, long_period, signal_period) -> np.ndarray:
    """Moving Average Convergence/Divergence

    Parameters
    ----------
    series : np.ndarray
        Input data.
    short_period : int
        Short period
    long_period : int
        Long period
    signal_period : int
        Signal period

    Returns
    -------
    np.ndarray
        macd_signal

    """
    return ti.macd(series, short_period, long_period, signal_period).macd_signal


def marketfi(high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Market Facilitation Index

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        marketfi

    """
    return ti.marketfi(high, low, volume)


def mass(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Mass Index

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        mass

    """
    return ti.mass(high, low, period)


def md(series: np.ndarray, period: int) -> np.ndarray:
    """Mean Deviation Over Period

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        md

    """
    return ti.md(series, period)


def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Money Flow Index

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.
    period : int


    Returns
    -------
    np.ndarray
        mfi

    """
    return ti.mfi(high, low, close, volume, period)


def natr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Normalized Average True Range

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        natr

    """
    return ti.natr(high, low, close, period)


def pfe(series: np.ndarray, period: int, ema_period: int) -> np.ndarray:
    """Polarized Fractal Efficiency

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter
    ema_period : int
        EMA period

    Returns
    -------
    np.ndarray
        pfe

    """
    return ti.pfe(series, period, ema_period)


def posc(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, ema_period: int) -> np.ndarray:
    """Projection Oscillator

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
        Period parameter
    ema_period : int
        EMA period

    Returns
    -------
    np.ndarray
        posc

    """
    return ti.posc(high, low, close, period, ema_period)


def ppo(series: np.ndarray, short_period: int, long_period: int) -> np.ndarray:
    """Percentage Price Oscillator

    Parameters
    ----------
    series : np.ndarray
        Input data.
    short_period : int
        Short period
    long_period : int
        Long period

    Returns
    -------
    np.ndarray
        ppo

    """
    return ti.ppo(series, short_period, long_period)


def qstick(open: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Qstick

    Parameters
    ----------
    open : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        qstick

    """
    return ti.qstick(open, close, period)


def rmi(series: np.ndarray, period: int, lookback_period: int) -> np.ndarray:
    """Relative Momentum Index

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter
    lookback_period : int
        Lookback period

    Returns
    -------
    np.ndarray
        rmi

    """
    return ti.rmi(series, period, lookback_period)


def roc(series: np.ndarray, period: int) -> np.ndarray:
    """Rate of Change

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        roc

    """
    return ti.roc(series, period)


def rocr(series: np.ndarray, period: int) -> np.ndarray:
    """Rate of Change Ratio

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        rocr

    """
    return ti.rocr(series, period)


def rsi(series: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        rsi

    """
    return ti.rsi(series, period)


def rvi(series: np.ndarray, ema_period: int, stddev_period: int) -> np.ndarray:
    """Relative Volatility Index

    Parameters
    ----------
    series : np.ndarray
        Input data.
    ema_period : int
        EMA period
    stddev_period : int
        stddev period

    Returns
    -------
    np.ndarray
        rvi

    """
    return ti.rvi(series, ema_period, stddev_period)


def smi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    q_period: int,
    r_period: int,
    s_period: int
) -> np.ndarray:
    """Stochastic Momentum Index

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    q_period : int
        Q period
    r_period : int
        R period
    s_period : int
        S period

    Returns
    -------
    np.ndarray
        smi

    """
    return ti.smi(high, low, close, q_period, r_period, s_period)


def stddev(series: np.ndarray, period: int) -> np.ndarray:
    """Standard Deviation Over Period

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        stddev

    """
    return ti.stddev(series, period)


def stderr(series: np.ndarray, period: int) -> np.ndarray:
    """Standard Error Over Period

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        stderr

    """
    return ti.stderr(series, period)


def stochrsi(series: np.ndarray, period: int) -> np.ndarray:
    """Stochastic RSI

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        stochrsi

    """
    return ti.stochrsi(series, period)


def trix(series: np.ndarray, period: int) -> np.ndarray:
    """Trix

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        trix

    """
    return ti.trix(series, period)


def tsi(series: np.ndarray, y_period: int, z_period: int) -> np.ndarray:
    """True Strength Index

    Parameters
    ----------
    series :
        Input data.
    y_period: int
        y period
    z_period: int
        z period

    Returns
    -------
    np.ndarray
        tsi

    """
    return ti.tsi(series, y_period, z_period)


def ultosc(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, short_period: int, medium_period: int, long_period: int
) -> np.ndarray:
    """Ultimate Oscillator

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    short_period : int
        Short period
    medium_period : int
        Medium period
    long_period : int
        Long period

    Returns
    -------
    np.ndarray
        ultosc

    """
    return ti.ultosc(high, low, close, short_period, medium_period, long_period)


def vosc(volume: np.ndarray, short_period: int, long_period: int) -> np.ndarray:
    """Volume Oscillator

    Parameters
    ----------
    volume : np.ndarray
        Input data.
    short_period : int
        Short period
    long_period : int
        Long period

    Returns
    -------
    np.ndarray
        vosc

    """
    return ti.vosc(volume, short_period, long_period)


def willr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Williams %R

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        willr

    """
    return ti.willr(high, low, close, period)


def kst_signal(series: np.ndarray, roc1, roc2, roc3, roc4, ma1, ma2, ma3, ma4) -> np.ndarray:
    """Know Sure Thing

    Parameters
    ----------
    series : np.ndarray
        Input series
    roc1 : np.ndarray
        Input series
    roc2 : np.ndarray
        Input series
    roc3 : np.ndarray
        Input series
    roc4 : np.ndarray
        Input series
    ma1 : np.ndarray
        Input series
    ma2 : np.ndarray
        Input series
    ma3 : np.ndarray
        Input series
    ma4 : np.ndarray
        Input series

    Returns
    -------
    np.ndarray
        kst_signal
    """
    return ti.kst(series, roc1, roc2, roc3, roc4, ma1, ma2, ma3, ma4).kst_signal


def minus_di(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Directional Indicator

    Parameters
    ----------
    high : np.ndarray
        Input series
    low : np.ndarray
        Input series
    close : np.ndarray
        Input series
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        minus_id
    """
    return ti.di(high, low, close, period).minus_di


def minus_dm(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Directional Movement

    Parameters
    ----------
    high : np.ndarray
        Input series
    low : np.ndarray
        Input series
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        minus_id
    """
    return ti.dm(high, low, period).minus_dm


def plus_di(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Directional Indicator

    Parameters
    ----------
    high : np.ndarray
        Input series
    low : np.ndarray
        Input series
    close : np.ndarray
        Input series
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        plus_di
    """
    return ti.di(high, low, close, period).plus_di


def plus_dm(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Directional Movement

    Parameters
    ----------
    high : np.ndarray
        Input series
    low : np.ndarray
        Input series
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        plus_dm
    """
    return ti.dm(high, low, period).plus_dm


def msw_sine(series: np.ndarray, period: int) -> np.ndarray:
    """Mesa Sine Wave

    Parameters
    ----------
    series : np.ndarray
        Input series
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        msw_sine
    """
    return ti.msw(series, period).msw_sine


def stoch_k(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period, k_slowing_period, d_period) -> np.ndarray:
    """Stochastic Oscillator

    Parameters
    ----------
    high : np.ndarray
        Input series
    low : np.ndarray
        Input series
    close : np.ndarray
        Input series
    k_period : int
        k period
    k_slowing_period : int
        k slowing period
    d_period : int
        d period

    Returns
    -------
    np.ndarray
        stoch_k
    """
    return ti.stoch(high, low, close, k_period, k_slowing_period, d_period).stoch_k


def stoch_d(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period, k_slowing_period, d_period) -> np.ndarray:
    """Stochastic Oscillator

    Parameters
    ----------
    high : np.ndarray
        Input series
    low : np.ndarray
        Input series
    close : np.ndarray
        Input series
    k_period : int
        k period
    k_slowing_period : int
        k slowing period
    d_period : int
        d period

    Returns
    -------
    np.ndarray
        stoch_d
    """
    return ti.stoch(high, low, close, k_period, k_slowing_period, d_period).stoch_d


def hurst(series: np.ndarray, period: int) -> np.ndarray:
    """Hurst Exponent Indicator

    Parameters
    ----------
    series : np.ndarray
        Input series
    period : int
        Period parameter

    Returns
    -------
    np.ndarray
        hurst
    """
    return ti.hurst(series, period).hurst

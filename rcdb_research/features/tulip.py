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


def arsi(series: np.ndarray, cycpart: int) -> np.ndarray:
    """Adaptive Relative Strength Index

    Parameters
    ----------
    series : np.ndarray
        Input data.
    cycpart : int

    Returns
    -------
    np.ndarray
        arsi
    """
    return ti.arsi(series, cycpart)


def bop(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Balance of Power

    Parameters
    ----------
    open : np.ndarray
        Input data.
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        bop
    """
    return ti.bop(open, high, low, close)


def cmi(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Choppy Market Indicator

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
        cmi
    """
    return ti.cmi(high, low, close, period)


def emsd(series: np.ndarray, period: int, ma_period: int) -> np.ndarray:
    """Exponential Moving Standard Deviation

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    ma_period : int

    Returns
    -------
    np.ndarray
        emsd
    """
    return ti.emsd(series, period, ma_period)


def er(series: np.ndarray, period: int) -> np.ndarray:
    """Efficiency Ratio

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        er
    """
    return ti.er(series, period)


def kvo(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, short_period: int,
        long_period: int) -> np.ndarray:
    """Klinger Volume Oscillator

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
    long_period : int

    Returns
    -------
    np.ndarray
        kvo
    """
    return ti.kvo(high, low, close, volume, short_period, long_period)


def mesastoch(series: np.ndarray, period: int, max_cycle_considered: int) -> np.ndarray:
    """MESA Stochastic (by John F. Ehlers)

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    max_cycle_considered : int

    Returns
    -------
    np.ndarray
        mesastoch
    """
    return ti.mesastoch(series, period, max_cycle_considered)


def msw_lead(series: np.ndarray, period: int) -> np.ndarray:
    """Mesa Sine Wave

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        msw_lead
    """
    return ti.msw(series, period).msw_lead


def nvi(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Negative Volume Index

    Parameters
    ----------
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        nvi
    """
    return ti.nvi(close, volume)


def pvi(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Positive Volume Index

    Parameters
    ----------
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        pvi
    """
    return ti.pvi(close, volume)


def var(series: np.ndarray, period: int) -> np.ndarray:
    """Variance Over Period

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        var
    """
    return ti.var(series, period)


def volatility(series: np.ndarray, period: int) -> np.ndarray:
    """Annualized Historical Volatility

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        volatility
    """
    return ti.volatility(series, period)


def abands_lower(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Acceleration Bands

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
        abands_lower
    """
    return ti.abands(high, low, close, period).abands_lower


def abands_upper(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Acceleration Bands

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
        abands_upper
    """
    return ti.abands(high, low, close, period).abands_upper


def abands_middle(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Acceleration Bands

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
        abands_middle
    """
    return ti.abands(high, low, close, period).abands_middle


def ad(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Accumulation/Distribution Line

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

    Returns
    -------
    np.ndarray
        ad
    """
    return ti.ad(high, low, close, volume)


def ahma(series: np.ndarray, period: int) -> np.ndarray:
    """Ahrens Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        ahma
    """
    return ti.ahma(series, period)


def aroon_down(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Aroon

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        aroon_down
    """
    return ti.aroon(high, low, period).aroon_down


def aroon_up(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Aroon

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        aroon_up
    """
    return ti.aroon(high, low, period).aroon_up


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range

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
        atr
    """
    return ti.atr(high, low, close, period)


def avgprice(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Average Price

    Parameters
    ----------
    open : np.ndarray
        Input data.
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        avgprice
    """
    return ti.avgprice(open, high, low, close)


def bbands_lower(series: np.ndarray, period: int, stddev: int) -> np.ndarray:
    """Bollinger Bands

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    stddev : int

    Returns
    -------
    np.ndarray
        bbands_lower
    """
    return ti.bbands(series, period, stddev).bbands_lower


def bbands_middle(series: np.ndarray, period: int, stddev: int) -> np.ndarray:
    """Bollinger Bands

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    stddev : int

    Returns
    -------
    np.ndarray
        bbands_middle
    """
    return ti.bbands(series, period, stddev).bbands_middle


def bbands_upper(series: np.ndarray, period: int, stddev: int) -> np.ndarray:
    """Bollinger Bands

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    stddev : int

    Returns
    -------
    np.ndarray
        bbands_upper
    """
    return ti.bbands(series, period, stddev).bbands_upper


def bf2(series: np.ndarray, period: int) -> np.ndarray:
    """Butterworth Filter - 2 Poles

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        bf2
    """
    return ti.bf2(series, period)


def bf3(series: np.ndarray, period: int) -> np.ndarray:
    """Butterworth Filter - 3 Poles

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        bf3
    """
    return ti.bf3(series, period)


def ce_high(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, coef: int) -> np.ndarray:
    """Chandelier Exit

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
    coef : int

    Returns
    -------
    np.ndarray
        ce_high
    """
    return ti.ce(high, low, close, period, coef).ce_high


def ce_low(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, coef: int) -> np.ndarray:
    """Chandelier Exit

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
    coef : int

    Returns
    -------
    np.ndarray
        ce_low
    """
    return ti.ce(high, low, close, period, coef).ce_low


def decay(series: np.ndarray, period: int) -> np.ndarray:
    """Linear Decay

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        decay
    """
    return ti.decay(series, period)


def dema(series: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        dema
    """
    return ti.dema(series, period)


def dwma(series: np.ndarray, period: int) -> np.ndarray:
    """Double Weighted Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        dwma
    """
    return ti.dwma(series, period)


def edcf(series: np.ndarray, length: int) -> np.ndarray:
    """Ehlers Distance Coefficient Filter

    Parameters
    ----------
    series : np.ndarray
        Input data.
    length : int

    Returns
    -------
    np.ndarray
        edcf
    """
    return ti.edcf(series, length)


def edecay(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Decay

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        edecay
    """
    return ti.edecay(series, period)


def ehma(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Hull Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        ehma
    """
    return ti.ehma(series, period)


def ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        ema
    """
    return ti.ema(series, period)


def evwma(close: np.ndarray, volume: np.ndarray, period: int, gamma: int) -> np.ndarray:
    """Elastic Volume Weighted Moving Average

    Parameters
    ----------
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.
    period : int
    gamma : int

    Returns
    -------
    np.ndarray
        evwma
    """
    return ti.evwma(close, volume, period, gamma)


def fi(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Force Index

    Parameters
    ----------
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        fi
    """
    return ti.fi(close, volume, period)


def frama(high: np.ndarray, low: np.ndarray, period: int, average_period: int) -> np.ndarray:
    """Fractal Adaptive Moving Average

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int
    average_period : int

    Returns
    -------
    np.ndarray
        frama
    """
    return ti.frama(high, low, period, average_period)


def gf1(series: np.ndarray, period: int) -> np.ndarray:
    """Gaussian Filter - 1 Pole

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        gf1
    """
    return ti.gf1(series, period)


def gf2(series: np.ndarray, period: int) -> np.ndarray:
    """Gaussian Filter - 2 Poles

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        gf2
    """
    return ti.gf2(series, period)


def gf3(series: np.ndarray, period: int) -> np.ndarray:
    """Gaussian Filter - 3 Poles

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        gf3
    """
    return ti.gf3(series, period)


def gf4(series: np.ndarray, period: int) -> np.ndarray:
    """Gaussian Filter - 4 Poles

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        gf4
    """
    return ti.gf4(series, period)


def hd(series: np.ndarray) -> np.ndarray:
    """Homodyne Discriminator

    Parameters
    ----------
    series : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        hd
    """
    return ti.hd(series)


def hf(series: np.ndarray, period: int, threshold: int) -> np.ndarray:
    """Hampel Filter

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    threshold : int

    Returns
    -------
    np.ndarray
        hf
    """
    return ti.hf(series, period, threshold)


def hfema(series: np.ndarray, ema_period: int, k: int, threshold: int) -> np.ndarray:
    """Hampel Filter on Exponential Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    ema_period : int
    k : int
    threshold : int

    Returns
    -------
    np.ndarray
        hfema
    """
    return ti.hfema(series, ema_period, k, threshold)


def hfsma(series: np.ndarray, sma_period: int, k: int, threshold: int) -> np.ndarray:
    """Hampel Filter on Simple Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    sma_period : int
    k : int
    threshold : int

    Returns
    -------
    np.ndarray
        hfsma
    """
    return ti.hfsma(series, sma_period, k, threshold)


def hma(series: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        hma
    """
    return ti.hma(series, period)


def hwma(series: np.ndarray, period: int) -> np.ndarray:
    """Henderson asymmetric filter

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        hwma
    """
    return ti.hwma(series, period)


def ichi_tenkan_sen(high: np.ndarray, low: np.ndarray, period9: int, period26: int, period52: int) -> np.ndarray:
    """Ichimoku

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period9 : int
    period26 : int
    period52 : int

    Returns
    -------
    np.ndarray
        ichi_tenkan_sen
    """
    return ti.ichi(high, low, period9, period26, period52).ichi_tenkan_sen


def ichi_kijun_sen(high: np.ndarray, low: np.ndarray, period9: int, period26: int, period52: int) -> np.ndarray:
    """Ichimoku

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period9 : int
    period26 : int
    period52 : int

    Returns
    -------
    np.ndarray
        ichi_kijun_sen
    """
    return ti.ichi(high, low, period9, period26, period52).ichi_kijun_sen


def ichi_senkou_span_A(high: np.ndarray, low: np.ndarray, period9: int, period26: int, period52: int) -> np.ndarray:
    """Ichimoku

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period9 : int
    period26 : int
    period52 : int

    Returns
    -------
    np.ndarray
        ichi_senkou_span_A
    """
    return ti.ichi(high, low, period9, period26, period52).ichi_senkou_span_A


def ichi_senkou_span_B(high: np.ndarray, low: np.ndarray, period9: int, period26: int, period52: int) -> np.ndarray:
    """Ichimoku

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period9 : int
    period26 : int
    period52 : int

    Returns
    -------
    np.ndarray
        ichi_senkou_span_B
    """
    return ti.ichi(high, low, period9, period26, period52).ichi_senkou_span_B


def idwma(series: np.ndarray, period: int, exponent: int) -> np.ndarray:
    """Inverse Distance Weighted Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    exponent : int

    Returns
    -------
    np.ndarray
        idwma
    """
    return ti.idwma(series, period, exponent)


def kama(series: np.ndarray, period: int) -> np.ndarray:
    """Kaufman Adaptive Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        kama
    """
    return ti.kama(series, period)


def kc_lower(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, multiple: int) -> np.ndarray:
    """Keltner Channel

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
    multiple : int

    Returns
    -------
    np.ndarray
        kc_lower
    """
    return ti.kc(high, low, close, period, multiple).kc_lower


def kc_middle(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, multiple: int) -> np.ndarray:
    """Keltner Channel

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
    multiple : int

    Returns
    -------
    np.ndarray
        kc_middle
    """
    return ti.kc(high, low, close, period, multiple).kc_middle


def kc_upper(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, multiple: int) -> np.ndarray:
    """Keltner Channel

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.
    period : int
    multiple : int

    Returns
    -------
    np.ndarray
        kc_upper
    """
    return ti.kc(high, low, close, period, multiple).kc_upper


def lag(series: np.ndarray, period: int) -> np.ndarray:
    """Lag

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        lag
    """
    return ti.lag(series, period)


def lf(series: np.ndarray, gamma: int) -> np.ndarray:
    """Laguerre Filter

    Parameters
    ----------
    series : np.ndarray
        Input data.
    gamma : int

    Returns
    -------
    np.ndarray
        lf
    """
    return ti.lf(series, gamma)


def linreg(series: np.ndarray, period: int) -> np.ndarray:
    """Linear Regression

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        linreg
    """
    return ti.linreg(series, period)


def linregintercept(series: np.ndarray, period: int) -> np.ndarray:
    """Linear Regression Intercept

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        linregintercept
    """
    return ti.linregintercept(series, period)


def lma(close: np.ndarray, period: int) -> np.ndarray:
    """Leo Moving Average

    Parameters
    ----------
    close : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        lma
    """
    return ti.lma(close, period)


def mama(series: np.ndarray, fastlimit: int, slowlimit: int) -> np.ndarray:
    """MESA Adaptive Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    fastlimit : int
    slowlimit : int

    Returns
    -------
    np.ndarray
        mama
    """
    return ti.mama(series, fastlimit, slowlimit).mama


def fama(series: np.ndarray, fastlimit: int, slowlimit: int) -> np.ndarray:
    """MESA Adaptive Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    fastlimit : int
    slowlimit : int

    Returns
    -------
    np.ndarray
        fama
    """
    return ti.mama(series, fastlimit, slowlimit).fama


def medprice(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Median Price

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        medprice
    """
    return ti.medprice(high, low)


def mgdyn(series: np.ndarray, N: int) -> np.ndarray:
    """McGinley Dynamic

    Parameters
    ----------
    series : np.ndarray
        Input data.
    N : int

    Returns
    -------
    np.ndarray
        mgdyn
    """
    return ti.mgdyn(series, N)


def mhlsma(series: np.ndarray, period: int, ma_period: int) -> np.ndarray:
    """Middle-High-Low Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    ma_period : int

    Returns
    -------
    np.ndarray
        mhlsma
    """
    return ti.mhlma(series, period, ma_period).mhlsma


def mhlema(series: np.ndarray, period: int, ma_period: int) -> np.ndarray:
    """Middle-High-Low Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    ma_period : int

    Returns
    -------
    np.ndarray
        mhlema
    """
    return ti.mhlma(series, period, ma_period).mhlema


def mom(series: np.ndarray, period: int) -> np.ndarray:
    """Momentum

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        mom
    """
    return ti.mom(series, period)


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On Balance Volume

    Parameters
    ----------
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        obv
    """
    return ti.obv(close, volume)


def pbands_lower(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Projection Bands

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
        pbands_lower
    """
    return ti.pbands(high, low, close, period).pbands_lower


def pbands_upper(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Projection Bands

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
        pbands_upper
    """
    return ti.pbands(high, low, close, period).pbands_upper


def pc_low(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Price Channel

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        pc_low
    """
    return ti.pc(high, low, period).pc_low


def pc_high(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Price Channel

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        pc_high
    """
    return ti.pc(high, low, period).pc_high


def psar(high: np.ndarray, low: np.ndarray, acceleration_factor_step: int,
         acceleration_factor_maximum: int) -> np.ndarray:
    """Parabolic SAR

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    acceleration_factor_step : int
    acceleration_factor_maximum : int

    Returns
    -------
    np.ndarray
        psar
    """
    return ti.psar(high, low, acceleration_factor_step, acceleration_factor_maximum)


def pvt(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Price Volume Trend

    Parameters
    ----------
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        pvt
    """
    return ti.pvt(close, volume)


def pwma(series: np.ndarray, period: int, power: int) -> np.ndarray:
    """Power Weighted Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    power : int

    Returns
    -------
    np.ndarray
        pwma
    """
    return ti.pwma(series, period, power)


def rema(series: np.ndarray, period: int, _lambda: int) -> np.ndarray:
    """Regularized Exponential Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    _lambda : int

    Returns
    -------
    np.ndarray
        rema
    """
    return ti.rema(series, period, _lambda)


def rmf(series: np.ndarray, critical_period: int, median_period: int) -> np.ndarray:
    """Recursive Median Filter

        Parameters
        ----------
        series : np.ndarray
            Input data.
        critical_period : int
        median_period : int

        Returns
        -------
        np.ndarray
            rmf
        """
    return ti.rmf(series, critical_period, median_period)


def rmta(series: np.ndarray, period: int, beta: int) -> np.ndarray:
    """Recursive Moving Trend Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    beta : int

    Returns
    -------
    np.ndarray
        rmta
    """
    return ti.rmta(series, period, beta)


def roof(series: np.ndarray) -> np.ndarray:
    """The Roofing Filter

    Parameters
    ----------
    series : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        roof
    """
    return ti.roof(series)


def shmma(series: np.ndarray, period: int) -> np.ndarray:
    """Sharp Modified Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        shmma
    """
    return ti.shmma(series, period)


def sma(series: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        sma
    """
    return ti.sma(series, period)


def ssmooth(series: np.ndarray) -> np.ndarray:
    """SuperSmoother Filter

    Parameters
    ----------
    series : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        ssmooth
    """
    return ti.ssmooth(series)


def swma(series: np.ndarray, period: int) -> np.ndarray:
    """Sine Weighted Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        swma
    """
    return ti.swma(series, period)


def t3(series: np.ndarray, period: int, v: int) -> np.ndarray:
    """T3 Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int
    v : int

    Returns
    -------
    np.ndarray
        t3
    """
    return ti.t3(series, period, v)


def tema(series: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        tema
    """
    return ti.tema(series, period)


def tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        tr
    """
    return ti.tr(high, low, close)


def trima(series: np.ndarray, period: int) -> np.ndarray:
    """Triangular Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        trima
    """
    return ti.trima(series, period)


def tsf(series: np.ndarray, period: int) -> np.ndarray:
    """Time Series Forecast

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        tsf
    """
    return ti.tsf(series, period)


def typprice(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Typical Price

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        typprice
    """
    return ti.typprice(high, low, close)


def vhf(series: np.ndarray, period: int) -> np.ndarray:
    """Vertical Horizontal Filter

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        vhf
    """
    return ti.vhf(series, period)


def vi_p(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Vortex Indicator

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
        vi_p
    """
    return ti.vi(high, low, close, period).vi_p


def vi_m(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Vortex Indicator

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
        vi_m
    """
    return ti.vi(high, low, close, period).vi_m


def vidya(series: np.ndarray, short_period: int, long_period: int, alpha: int) -> np.ndarray:
    """Variable Index Dynamic Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    short_period : int
    long_period : int
    alpha : int

    Returns
    -------
    np.ndarray
        vidya
    """
    return ti.vidya(series, short_period, long_period, alpha)


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume Weighted Average Price

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
        vwap
    """
    return ti.vwap(high, low, close, volume, period)


def vwma(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume Weighted Moving Average

    Parameters
    ----------
    close : np.ndarray
        Input data.
    volume : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        vwma
    """
    return ti.vwma(close, volume, period)


def wad(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Williams Accumulation/Distribution

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        wad
    """
    return ti.wad(high, low, close)


def wcprice(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Weighted Close Price

    Parameters
    ----------
    high : np.ndarray
        Input data.
    low : np.ndarray
        Input data.
    close : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        wcprice
    """
    return ti.wcprice(high, low, close)


def wilders(series: np.ndarray, period: int) -> np.ndarray:
    """Wilders Smoothing

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        wilders
    """
    return ti.wilders(series, period)


def wma(series: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        wma
    """
    return ti.wma(series, period)


def zlema(series: np.ndarray, period: int) -> np.ndarray:
    """Zero-Lag Exponential Moving Average

    Parameters
    ----------
    series : np.ndarray
        Input data.
    period : int

    Returns
    -------
    np.ndarray
        zlema
    """
    return ti.zlema(series, period)

import joblib
import numpy as np
import pandas as pd


BAR_CLOSE = 0
BAR_TRGT = 1
BAR_PT = 2
BAR_SL = 3
BAR_PT_PCT = 4
BAR_SL_PCT = 5
BAR_NUM = 6
BAR_ACTIVE_BARS_COUNTER = 7
BAR_TIMESTAMP_BARRIER = 8


def __build_triple_barrier_labels(
        close_arr: np.ndarray,
        timestamp_arr: np.ndarray,
        trgt_arr: np.ndarray,
        min_ret_arr: np.ndarray,
        t_events_arr: np.ndarray,
        vertical_barrier_time_arr: np.ndarray = None,
        pt_sl=(1, 1),
        calc_metrics=False,
        offset: int = 0,
        limit: int = None
):
    """
    Calc triple barrier

    :param close_arr:
    :param timestamp_arr:
    :param trgt_arr:
    :param min_ret_arr:
    :param t_events_arr:
    :param vertical_barrier_time_arr:
    :param pt_sl:
    :param calc_metrics:
    :param offset:
    :param limit:
    :return:
    """

    arr_size = close_arr.size
    if limit is None:
        limit = arr_size - offset
    job_range = range(offset, arr_size)

    use_vertical_barrier_time = vertical_barrier_time_arr is not None
    default_bin = np.nan
    if use_vertical_barrier_time:
        default_bin = 0

    # active labels storage, keys - ints (bar number)
    active_labels = dict()

    # triple barrier arrays
    bars__ret = np.empty(arr_size) * np.nan
    bars__trgt = np.empty(arr_size) * np.nan
    bars__pt = np.empty(arr_size) * np.nan
    bars__sl = np.empty(arr_size) * np.nan
    bars__bin = np.empty(arr_size) * np.nan
    bars__t1 = np.empty(arr_size) * np.nan

    # metrics
    bars__active_count = np.empty(arr_size) * np.nan if calc_metrics else None
    bars__uniqueness_mean = np.empty(arr_size) * np.nan if calc_metrics else None
    bars__uniqueness_mean_harm = np.empty(arr_size) * np.nan if calc_metrics else None

    for i in job_range:
        if i > offset + limit and len(active_labels) == 0:
            break

        close = close_arr[i]
        timestamp = timestamp_arr[i]
        target = trgt_arr[i]
        min_ret = abs(min_ret_arr[i])
        t_event = t_events_arr[i]

        if np.isnan(target) or target < min_ret:
            continue

        if use_vertical_barrier_time:
            vertical_barrier_time = vertical_barrier_time_arr[i]

        pt_pct = pt_sl[0] * target
        sl_pct = -abs(pt_sl[1]) * target

        if (not np.isnan(t_event) and t_event) and i <= offset + limit:
            # BAR_CLOSE = 0
            # BAR_TRGT = 1
            # BAR_PT = 2
            # BAR_SL = 3
            # BAR_PT_PCT = 4
            # BAR_SL_PCT = 5
            # BAR_NUM = 6
            # BAR_ACTIVE_BARS_COUNTER = 7
            # BAR_TIMESTAMP_BARRIER = 8

            active_labels[i] = [close, target, pt_sl[0], pt_sl[1], pt_pct, sl_pct, i, [], None]

            if use_vertical_barrier_time:
                active_labels[i][BAR_TIMESTAMP_BARRIER] = vertical_barrier_time

        if calc_metrics:
            # collect active bars metrics
            active_bars_count = len(active_labels)
            for j in range(active_bars_count):
                active_labels[j][BAR_ACTIVE_BARS_COUNTER].append(active_bars_count)

        labels_to_delete = []

        for bar_num, active_label in active_labels.items():
            is_vertical_hit = use_vertical_barrier_time and timestamp >= active_label[BAR_TIMESTAMP_BARRIER]

            # price percent change since label start until now
            pct_change = close / active_label[BAR_CLOSE] - 1
            is_sl_hit = pct_change <= active_label[BAR_SL_PCT]
            is_tp_hit = pct_change >= active_label[BAR_PT_PCT]

            if not (is_sl_hit or is_tp_hit or is_vertical_hit):
                # neither of barriers were touched
                continue
            # barrier was touched -> closing current label

            bar_num = active_label[BAR_NUM]

            bars__ret[bar_num] = pct_change
            bars__trgt[bar_num] = active_label[BAR_TRGT]
            bars__bin[bar_num] = 1 if is_tp_hit else -1 if is_sl_hit else 0 if is_vertical_hit else default_bin

            bars__pt[bar_num] = active_label[BAR_PT]
            bars__sl[bar_num] = active_label[BAR_SL]
            bars__t1[bar_num] = timestamp

            labels_to_delete.append(bar_num)

            if calc_metrics:
                bars__active_count[bar_num] = i - active_label[BAR_NUM]

                unq_count_arr = np.array(active_label[3])
                bars__uniqueness_mean[bar_num] = unq_count_arr.mean()

                unq_count_harm_arr = (1 / unq_count_arr).sum() / (unq_count_arr.size - 1)
                bars__uniqueness_mean_harm[bar_num] = unq_count_harm_arr

        for k in labels_to_delete:
            # remove closed labels
            del active_labels[k]

    offset_limit = offset + limit
    result = (
        # data
        [
            bars__t1[offset:offset_limit],
            bars__trgt[offset:offset_limit],
            bars__pt[offset:offset_limit],
            bars__sl[offset:offset_limit],
            bars__ret[offset:offset_limit],
            bars__bin[offset:offset_limit]
        ],
        # columns
        [
            't1', 'trgt', 'pt', 'sl', 'ret', 'bin'
        ],
        # index
        timestamp_arr[offset:offset_limit]
    )

    if calc_metrics:
        # add metrics to result
        result[0] += [
            bars__active_count[offset:offset_limit],
            bars__uniqueness_mean[offset:offset_limit],
            bars__uniqueness_mean_harm[offset:offset_limit]]

        result[1] += ['active_count', 'uniqueness_mean', 'uniqueness_mean_harm']

    return result


def result_to_pd(results):
    d = {}
    for i in range(len(results[0])):
        d[results[1][i]] = results[0][i]
    df = pd.DataFrame(d, index=results[2])
    return df


def get_events_rcdb(df_data, pt_sl=(1, 1), n_jobs=1):
    """
    Calculate triple barrier labels in parallel (wrapper for single thread `__build_triple_barrier_labels`)
    :param df_data:
    :param pt_sl:
    :param n_jobs:
    :return:
    """
    if n_jobs == -1:
        n_jobs = joblib.cpu_count()

    dataset_size = df_data.shape[0]
    batch_size = dataset_size // n_jobs + 1

    is_vertical_barrier_time = "vertical_barrier_time" in list(df_data.columns)

    jobs = []
    for offset in range(0, dataset_size, batch_size):
        jobs.append(joblib.delayed(__build_triple_barrier_labels)(
            close_arr=df_data.close.values[offset:],
            timestamp_arr=df_data.index.values[offset:],
            trgt_arr=df_data.target.values[offset:],
            min_ret_arr=df_data.min_ret.values[offset:],
            t_events_arr=df_data.t_events.values[offset:],
            vertical_barrier_time_arr=df_data.vertical_barrier_time.values[offset:]
            if is_vertical_barrier_time else None,
            pt_sl=pt_sl,
            offset=0,
            limit=batch_size,
            calc_metrics=False
        ))

    results = joblib.Parallel(n_jobs=n_jobs)(jobs)

    df = result_to_pd(results[0])
    for i in range(1, len(results)):
        df = df.append(result_to_pd(results[i]))

    if 'side_prediction' in list(df_data.columns):
        df['side'] = np.where(df_data['side_prediction'] == df['bin'], 1, 0)

    df['t1'] = pd.to_datetime(df['t1'])

    return df


def get_events_hnt_proxy(
        close: pd.Series, t_events: pd.DatetimeIndex, target: pd.Series,
        pt_sl: tuple = (1, 1), min_ret: float = 0, num_threads: int = 1,
        vertical_barrier_times=False, side_prediction=None):
    """
    Calculate triple barrier labels (H&T compatible interface)

    :param close:
    :param t_events:
    :param target:
    :param pt_sl:
    :param min_ret:
    :param num_threads:
    :param vertical_barrier_times:
    :param side_prediction:
    :return:
    """

    df = pd.DataFrame()
    df['target'] = target
    df['t_events'] = False
    df['close'] = close
    df['min_ret'] = min_ret
    df.loc[df.index.isin(t_events), 't_events'] = True

    if isinstance(vertical_barrier_times, pd.Series):
        df['vertical_barrier_time'] = vertical_barrier_times
    if isinstance(side_prediction, pd.Series):
        df['side_prediction'] = side_prediction

    df = get_events_rcdb(df, pt_sl=pt_sl, n_jobs=num_threads)
    cols = ['t1', 'trgt', 'pt', 'sl']
    if side_prediction is not None:
        cols.append('side')
    return df[~df['t1'].isna()][cols]

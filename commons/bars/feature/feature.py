import pandas as pd
import numpy as np
import uuid

def feature(
            df: pd.DataFrame,
            column_name: str,
            aggregate: dict = {"open": 'first', "high": 'max', "low": 'min', "close": 'last',
                               "volume_buy": 'sum', "volume_sell": 'sum', "volume_quote_sell": 'sum',
                               "volume_quote_buy": 'sum', "ticks_sell": 'sum',
                               "ticks_buy": 'sum'},
            aggregate_default='first'
        ):
    # drop_first_bar = np.isnan(df[column_name].values[0])

    # prevent bugs with default mutable dict
    aggregate = aggregate.copy()

    # save index
    index_tmp_name = str(uuid.uuid4())
    index_prev_name = df.index.name

    columns = list(df.columns) + [index_tmp_name]
    df[index_tmp_name] = df.index

    # tmp column for aggregation
    agg_id_name = str(uuid.uuid4())
    df[agg_id_name] = df[column_name].cumsum() # np.where(df[column_name] != df[column_name].shift(1), 1, 0).cumsum()
    tmp = df[agg_id_name].values
    tmp[0] = tmp[1]
    df[agg_id_name] = tmp

    # apply default aggregation
    cols_exists = []
    for col in df.columns:
        cols_exists.append(col)
        if col not in aggregate:
            aggregate[col] = aggregate_default

    for col in list(aggregate.keys()):
        if col not in cols_exists:
            del aggregate[col]

    # aggregate
    df_new = df.groupby([agg_id_name]).agg(aggregate)[columns]
    df.drop([agg_id_name, index_tmp_name], axis=1, inplace=True)

    # return original index
    df_new = df_new.set_index(index_tmp_name)
    df_new.index.rename(index_prev_name, inplace=True)

    # if drop_first_bar:
    #     df_new = df_new[1:]

    return df_new

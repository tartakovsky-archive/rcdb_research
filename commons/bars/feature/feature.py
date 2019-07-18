import os
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
    # save index
    index_tmp_name = str(uuid.uuid4())
    index_prev_name = df.index.name
    df[index_tmp_name] = df.index
    columns = list(df.columns)

    # tmp column for aggregation
    agg_id_name = str(uuid.uuid4())
    df[agg_id_name] = (df[column_name] != df[column_name].shift(1)).cumsum()

    # apply default aggregation
    for col in df.columns:
        if col not in aggregate:
            aggregate[col] = aggregate_default

    # aggregate
    df_new = df.groupby([agg_id_name]).agg(aggregate)[columns]
    df.drop([agg_id_name, index_tmp_name], axis=1, inplace=True)

    # return original index
    df_new = df_new.set_index(index_tmp_name)
    df_new.index.rename(index_prev_name, inplace=True)

    return df_new

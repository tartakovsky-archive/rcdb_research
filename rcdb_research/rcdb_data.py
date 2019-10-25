from __future__ import annotations

import os
import io
import urllib
import logging
import multiprocessing
from collections import namedtuple
from typing import Optional

import numpy as np
import pandas as pd
import retrying
import requests

from rcdb_research import utils


class RcdbData:
    """
    Class which helps to access ohlcv data
    """
    _instance: RcdbData = None  # noqa

    OHLCVConfig = namedtuple(
        "OHLCVConfig", ["base", "quote", "exchange", "timeframe", "start", "end", "is_whole_period"]
    )

    LOCAL_CACHE_FILENAME = "{exchange}__{base}_{quote}_{timeframe}.hdf"

    def __init__(self, ohlcv_api_url: str, local_cache_path: str):
        """
        :param str ohlcv_api_url: ohlcv ip url
        :param str local_cache_path: path to local cahce dir
        """
        if ohlcv_api_url is None:
            raise ValueError("ohlcv_api_url doesn`t provided")

        if local_cache_path is None:
            raise ValueError("Provide cache_path parameter")

        self.ohlcv_api_url = ohlcv_api_url
        self.local_cache_path = local_cache_path
        self._local_cache = {}

    def fetch_df(self, ohlcv_config: RcdbData.OHLCVConfig) -> Optional[pd.DataFrame]:  # noqa
        """
        Fetch dataframe with ohlcv data
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: dataframe with ohlcv data
        :rtype: None or pd.DataFrame
        """
        for fetch_func in [
            self.fetch_from_local_file_cache,
            self.fetch_remote
        ]:
            logging.debug(f"Try fetch by {fetch_func.__name__}")
            df = fetch_func(ohlcv_config)

            if df is not None:
                logging.debug(f"Success fetch by {fetch_func.__name__}")
                break
        else:
            return None

        if ohlcv_config.is_whole_period:
            return df

        if ohlcv_config.start is None:
            return df[df.index < ohlcv_config.end]

        if ohlcv_config.end is None:
            return df[df.index >= ohlcv_config.start]

        return df[(df.index >= ohlcv_config.start) & (df.index < ohlcv_config.end)]

    def get_local_path(self, ohlcv_config: RcdbData.OHLCVConfig) -> str:  # noqa
        """
        Format local path template. Concatenate paths to local cache
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: path to local cache file
        :rtype: str
        """
        return os.path.join(
            self.local_cache_path,
            self.LOCAL_CACHE_FILENAME.format(
                **ohlcv_config._asdict()
            )
        )

    def get_ohlcv_url(self, ohlcv_config: RcdbData.OHLCVConfig) -> str:  # noqa
        """
        Format gcs path template
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: path to gcs file
        :rtype: str
        """
        get_query = urllib.parse.urlencode(
            dict(
                exchange=ohlcv_config.exchange,
                symbol=(ohlcv_config.base + ohlcv_config.quote).lower(),
                timeframe=ohlcv_config.timeframe
            )
        )
        return f"{self.ohlcv_api_url}?{get_query}"

    def fetch_from_local_file_cache(self, ohlcv_config: RcdbData.OHLCVConfig) -> Optional[pd.DataFrame]:  # noqa
        """
        Check file with df bytes and read df from it
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: dataframe with ohlcv data
        :rtype: pd.DataFrame or None
        """
        logging.debug(f"-> fetch_from_local_file_cache")
        local_path = self.get_local_path(ohlcv_config)
        if not os.path.exists(local_path):
            return None

        with open(local_path, "rb") as file:
            df = utils.get_df_from_hdf_bytes(file.read())

        return df

    @retrying.retry(
        retry_on_exception=lambda e: isinstance(e, requests.RequestException),
        wait_exponential_multiplier=1000,
        stop_max_attempt_number=3
    )
    def fetch_remote(self, ohlcv_config: RcdbData.OHLCVConfig) -> Optional[pd.DataFrame]:  # noqa
        """
        Try to download df data.
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: dataframe with ohlcv data
        :rtype: pd.DataFrame or None
        """
        logging.warning(f"Fetch remote {ohlcv_config}")
        resp = requests.get(
            self.get_ohlcv_url(ohlcv_config)
        )
        resp.raise_for_status()

        urls = resp.json()
        logging.warning(f"Start fetching {urls}")

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            df = pd.concat(pool.map(self.remote_read_df, urls))

        # df preparing
        logging.warning("Sort df")
        df.sort_index(inplace=True)

        self._cache_write_local_file(ohlcv_config, df)
        return df

    def _cache_write_local_file(self, ohlcv_config: RcdbData.OHLCVConfig, df: pd.DataFrame):  # noqa
        """
        Write df to local file
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :param df: source df
        :return:
        """
        local_path = self.get_local_path(ohlcv_config)
        with open(local_path, "wb") as file:
            file.write(
                utils.store_df_to_hdf_bytes(df)
            )

    @classmethod
    def fetch(
        cls,
        base: str,
        quote: str,
        exchange: str,
        timeframe: str = '1s',
        start: Optional[str] = None,
        end: Optional[str] = None,
        ohlcv_api_url: Optional[str] = None,
        local_cache_path: str = "../data",
    ) -> Optional[pd.DataFrame]:
        """
        if OHLCV instance don`t exists then init it. Fetch df by parameters.
        :param str base: base currency
        :param str quote: quote currency
        :param str exchange: excange name
        :param str timeframe: ohlc timeframe
        :param str start: start timestamp filter
        :param str end: end timestamp filter
        :param str local_cache_path: path lo local cache dir
        :param str ohlcv_api_url: currency service url. Default None. if default is try get from env var OHLCV_API_URL
        :return: dataframe with olcv
        :rtype: pd.DataFrame or None
        """
        if not os.path.exists(local_cache_path):
            os.makedirs(local_cache_path)

        ohlcv_config = cls.OHLCVConfig(
            base, quote, exchange, timeframe, start, end, is_whole_period=(not start and not end))

        logging.debug(f"Fetch by params: {ohlcv_config._asdict()}")

        if cls._instance is None:
            cls._instance = RcdbData(
                ohlcv_api_url or os.environ.get("OHLCV_API_URL"),
                local_cache_path
            )
            logging.debug("OHLCV -> inited")

        return cls._instance.fetch_df(ohlcv_config)

    @classmethod
    def clean_up(cls):
        """
        Remove OHLCV instance
        :return:
        """
        cls._instance = None

    ##########
    # Dataframe validation
    ##########
    @staticmethod
    def missed_columns(df: pd.DataFrame) -> list:
        """
        Check dataframe columns
        :param df:  input dataframe
        :return: list of missed columns
        """
        required_columns = {
            'open', 'high', 'low', 'close', 'volume', 'volume_buy', 'volume_sell',
            'volume_quote', 'volume_quote_buy', 'volume_quote_sell', 'ticks', 'ticks_buy', 'ticks_sell'
        }
        return list((required_columns & set(df.columns)) ^ required_columns)

    @staticmethod
    def check_consistency(df) -> bool:
        return all(dtype in [float, int] for dtype in df.dtypes) and np.isfinite(df).all().all()

    @staticmethod
    def consistency_info(df, verbose=False) -> tuple:
        missing = df.isnull().sum().where(lambda x: x > 0).dropna()
        infs = np.isinf(df).sum().where(lambda x: x > 0).dropna()
        duplicates = df.T[df.T.duplicated()].T.columns

        is_consistent = len(missing) == 0 and len(infs) == 0 and len(duplicates) == 0

        if verbose:
            print(f'\nColumns with missing values:\n {missing}')
            print(f'\nColumns with inf values:\n {infs}')
            print(f'\nDuplicated columns:\n {duplicates}')

        return (is_consistent, missing, infs, duplicates)

    @staticmethod
    def remote_read_df(url):
        logging.warning(f'Fetch shard {url}')
        resp = requests.get(url)
        resp.raise_for_status()

        logging.warning(f'Prepare {url}')
        df = pd.read_csv(
            io.BytesIO(resp.content),
            index_col="timestamp",
            compression="gzip"
        )
        df.index = pd.to_datetime(df.index, unit='s')
        logging.warning(f'{url} finished')
        return df

from __future__ import annotations

import os
import logging
from collections import namedtuple
from typing import Optional

import pandas as pd
import retrying
import requests

from commons import utils


class OHLCV:
    """
    Class which helps to access ohlcv data
    """
    _instance: OHLCV = None  # noqa

    OHLCVConfig = namedtuple(
        "OHLCVConfig", ["base", "quote", "exchange", "timeframe", "start", "end", "is_whole_period"]
    )

    REMOTE_PATH_TEMPLATE = "{exchange}/{base}/{quote}.hdf"
    LOCAL_CACHE_FILENAME = "{exchange}__{base}_{quote}.hdf"

    def __init__(self, ohlcv_api_url: str, local_cache_path: str):
        """
        :param str ohlcv_api_url: ohlcv ip url
        :param str local_cache_path: path to local cahce dir
        """
        assert ohlcv_api_url is not None, "ohlcv_api_url doesn`t provided"
        assert local_cache_path is not None, "Provide cache_path parameter"
        self.ohlcv_api_url = ohlcv_api_url
        self.local_cache_path = local_cache_path
        self._local_cache = {}

    def fetch_df(self, ohlcv_config: OHLCV.OHLCVConfig) -> Optional[pd.DataFrame]:  # noqa
        """
        Fetch dataframe with ohlcv data
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: dataframe with ohlcv data
        :rtype: None or pd.DataFrame
        """
        for fetch_func in [
            self.fetch_from_local_cache,
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

        return df[(df.index >= ohlcv_config.start) & (df.index < ohlcv_config.end)]

    def get_local_path(self, ohlcv_config: OHLCV.OHLCVConfig) -> str:  # noqa
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

    def get_ohlcv_url(self, ohlcv_config: OHLCV.OHLCVConfig) -> str:  # noqa
        """
        Format gcs path template
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: path to gcs file
        :rtype: str
        """
        # path = self.REMOTE_PATH_TEMPLATE.format(
        #     **ohlcv_config._asdict()
        # )
        exchange = getattr(ohlcv_config, 'exchange')
        symbol = (getattr(ohlcv_config, 'base') + getattr(ohlcv_config, 'quote')).lower()
        return f"{self.ohlcv_api_url}?exchange={exchange}&symbol={symbol}"

    def fetch_from_local_cache(self, ohlcv_config: OHLCV.OHLCVConfig) -> Optional[pd.DataFrame]:  # noqa
        """
        Returns dataframe from dict
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: dataframe with ohlcv data
        :rtype: pd.DataFrame or None
        """
        return self._local_cache.get(
            self.get_local_path(ohlcv_config)
        )

    def fetch_from_local_file_cache(self, ohlcv_config: OHLCV.OHLCVConfig) -> Optional[pd.DataFrame]:  # noqa
        """
        Check file with df bytes and read df from it
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: dataframe with ohlcv data
        :rtype: pd.DataFrame or None
        """
        local_path = self.get_local_path(ohlcv_config)
        if not os.path.exists(local_path):
            return None

        with open(local_path, "rb") as file:
            df = utils.get_df_from_hdf_bytes(file.read())

        self._cache_write_local(ohlcv_config, df)
        return df

    @retrying.retry(
        retry_on_exception=lambda e: isinstance(e, requests.RequestException),
        wait_exponential_multiplier=1000,
        stop_max_attempt_number=3
    )
    def fetch_remote(self, ohlcv_config: OHLCV.OHLCVConfig) -> Optional[pd.DataFrame]:  # noqa
        """
        Try to download df data.
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :return: dataframe with ohlcv data
        :rtype: pd.DataFrame or None
        """
        resp = requests.get(
            self.get_ohlcv_url(ohlcv_config)
        )
        resp.raise_for_status()

        df = utils.get_df_from_hdf_bytes(resp.content)
        self._cache_write(ohlcv_config, df)
        return df

    def _cache_write_local(self, ohlcv_config: OHLCV.OHLCVConfig, df: pd.DataFrame):  # noqa
        """
        Write df to dict cache
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :param df: source df
        :return:
        """
        self._local_cache[self.get_local_path(ohlcv_config)] = df

    def _cache_write(self, ohlcv_config: OHLCV.OHLCVConfig, df: pd.DataFrame):  # noqa
        """
        Write df to cache
        :param OHLCV.OHLCVConfig ohlcv_config: ohlcv data config
        :param df: source df
        :return:
        """
        self._cache_write_local(ohlcv_config, df)
        self._cache_write_local_file(ohlcv_config, df)

    def _cache_write_local_file(self, ohlcv_config: OHLCV.OHLCVConfig, df: pd.DataFrame):  # noqa
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
        timeframe: Optional[str] = None,  # deprecated?
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
            cls._instance = OHLCV(
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

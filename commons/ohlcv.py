import io
import os
import logging
from collections import namedtuple

import pandas as pd
from google.oauth2.service_account import Credentials
from google.cloud import storage

from lib import utils


class OHLCV:
    _instance = None
    bucket = None

    OHLCVConfig = namedtuple(
        "OHLCVConfig", ["base", "quote", "exchange", "timeframe", "start", "end", "is_whole_period"]
    )

    GCS_CACHE_FILENAME = "{exchange}/{base}/{quote}/{timeframe}.hdf"
    LOCAL_CACHE_FILENAME = GCS_CACHE_FILENAME

    def __init__(self, bucket_name):
        self.google_creds = Credentials.from_service_account_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        self.google_cloud_project = self.google_creds.project_id

        self._local_cache = {}
        self.bucket_name = bucket_name

        if bucket_name is None:
            logging.warning("Bucket name doesn`t provided. Operation with gcs disabled")
        else:
            self.bucket = storage.Client().get_bucket(bucket_name)

    def fetch_df(self, ohlcv_config):
        logging.debug("read from local cache")
        df = self.fetch_from_local_cache(ohlcv_config)

        if df.empty:
            logging.debug("local cache missed")
            logging.debug("read from gcs cache")
            df = self.fetch_from_gcs(ohlcv_config)

            if not df.empty:
                self._cache_write(ohlcv_config, df, gcs=False)

            else:
                logging.debug("gcs cache -> missed")
                logging.debug("read from bq")
                df = self.fetch_from_bq(ohlcv_config)

                if not df.empty:
                    self._cache_write(ohlcv_config, df)

        if ohlcv_config.is_whole_period:
            return df

        return df[df.index >= ohlcv_config.start & df.index < ohlcv_config.end]

    def fetch_from_local_cache(self, ohlcv_config):
        """
        :param OHLCV.OHLCVConfig ohlcv_config:
        :return:
        """
        return self._local_cache.get(
            self.LOCAL_CACHE_FILENAME.format(**ohlcv_config._asdict()),
            pd.DataFrame([])
        )

    def fetch_from_gcs(self, ohlcv_config):
        """
        :param OHLCV.OHLCVConfig ohlcv_config:
        :return:
        """
        if self.bucket is None:
            logging.warning(f"Cache read warning. Bucket name does`nt provided")
            return pd.DataFrame([])

        buffer = io.BytesIO()
        try:
            blob = self.bucket.get_blob(self.GCS_CACHE_FILENAME.format(**ohlcv_config._asdict()))
            blob.download_to_file(buffer)
            return utils.get_df_from_hdf_bytes(buffer.getvalue())

        except Exception:
            return pd.DataFrame([])

    def fetch_from_bq(self, ohlcv_config):
        table_name = f"{self.google_cloud_project}.{ohlcv_config.exchange}." \
                     f"{ohlcv_config.base}_{ohlcv_config.quote}__{ohlcv_config.timeframe}"

        query = f"""
            #standardsql
    
            SELECT
                TIMESTAMP_SECONDS(UNIX_SECONDS(timestamp)) timestamp, open, high, low, close, volume
            FROM `{table_name}`
            ORDER BY timestamp
        """

        df = pd.read_gbq(
            query,
            credentials=self.google_creds,
            index_col="timestamp",
            dialect="standard"
        )
        return df

    def _cache_write(self, ohlcv_config, df, gcs=True):
        cache_key = self.LOCAL_CACHE_FILENAME.format(**ohlcv_config._asdict())
        self._local_cache[cache_key] = df

        if gcs:
            self._cache_write_gcs(ohlcv_config, df)

    def _cache_write_gcs(self, ohlcv_config, df):
        if self.bucket is None:
            logging.warning(f"Cache write warning. Bucket name does`nt provided")
            return

        blob = storage.Blob(self.GCS_CACHE_FILENAME.format(**ohlcv_config._asdict()), self.bucket)

        blob.upload_from_file(
            io.BytesIO(utils.store_df_to_hdf_bytes(df))
        )

    @staticmethod
    def fetch(base, quote, exchange, timeframe, start=None, end=None, bucket_name=None):
        ohlcv_config = OHLCV.OHLCVConfig(
            base, quote, exchange, timeframe, start, end, is_whole_period=(not start and not end))

        logging.debug(f"Fetch by params: {ohlcv_config._asdict()}")
        if OHLCV._instance is None:
            OHLCV._instance = OHLCV(bucket_name)
            logging.debug("OHLCV -> inited")

        return OHLCV._instance.fetch_df(ohlcv_config)

    @staticmethod
    def clean_up():
        OHLCV._instance = None

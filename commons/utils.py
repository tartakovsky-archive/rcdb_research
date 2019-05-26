import pandas as pd


def store_df_to_hdf_bytes(df: pd.DataFrame, key: str = "table") -> bytes:
    with pd.HDFStore(
        "hdfs.tmp",
        mode="w",
        driver_core_backing_store=0,
        driver="H5FD_CORE"
    ) as out:

        out[key] = df
        return out._handle.get_file_image()


def get_df_from_hdf_bytes(hdf_bytes: bytes, key: str = "table") -> pd.DataFrame:
    with pd.HDFStore(
        "hdfs.tmp",
        mode="r",
        driver_core_backing_store=0,
        driver_core_image=hdf_bytes,
        driver="H5FD_CORE",
    ) as storage:
        return storage[key]

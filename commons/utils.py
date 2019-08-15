import subprocess
import os
import pandas as pd
import numpy as np
import joblib
import uuid
from typing import List, Dict, Callable
from sklearn.model_selection import ParameterGrid


# from dask.distributed import Client
# client = Client(processes=False)


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


def calc_all_parallel(func_calls_tasks, n_jobs=-1):
    """
    Helper to merge and process all feature calculation across multiple modules in one processes pull
    :param func_calls_tasks: dict(prefix1=function_calls_list1, prefix2=function_calls_list2, ...)
    :param n_jobs: process count to spawn, -1 - use all CPUs, 1 - no parallel computing code is used at all
    :return:
    """

    if n_jobs == -1:
        try:
            # Try to determine CPU count and set jobs equal to CPUs (optimal case)
            n_jobs = joblib.cpu_count()
        except Exception:
            pass

    prefix_list = []
    func_calls_list = []
    for k, v in func_calls_tasks.items():
        prefix_list += [k] * len(v)
        func_calls_list += v

    result = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(func)(*args) for [func, args] in func_calls_list)

    result_dict = dict()
    for i in range(0, len(result)):
        fn_name = func_calls_list[i][0].__name__
        fn_args = func_calls_list[i][1]
        args_to_str = ""

        if fn_args:
            params = [f"{v}" for v in fn_args if type(v) in [float, int, str]]
            if params:
                args_to_str = "__" + "__".join(params)

        result_name = f"{fn_name}{args_to_str}"
        result_name = f"{prefix_list[i]}__{result_name}"

        result_dict[result_name] = result[i]

    return pd.DataFrame(result_dict)


def symlog2_(x):
    C = 0  # parameter
    return np.sign(x) * (np.log2(1 + abs(x) / (10 ** C)))


def custom_transform(data, p1: int, p2: str):
    return data


class TransformObj:
    def __init__(self, transform_name: str, fn: Callable, **kwargs):
        self.transform_name = transform_name
        self.__fn = fn

        if not kwargs:
            kwargs = {}
        self.kwargs = kwargs

    def apply(self, data):
        return self.__fn(data, **self.kwargs)

    def get_name(self):
        return f"{self.transform_name}{kwargs_to_str(self.kwargs)}"


class Transforms:
    @staticmethod
    def symlog():
        return TransformObj("symlog", symlog2_)

    @staticmethod
    def custom_transform(p1: int, p2: str):
        return TransformObj("custom_transform", custom_transform, p1=p1, p2=p2)


class Col:
    def __init__(self, name):
        self.name = name
        self.transforms = []

    def t(self, transforms: List[TransformObj]):
        self.transforms += transforms
        return self

    def get_value(self, df):
        d = None
        if self.name == "index":
            if isinstance(df.index, pd.DatetimeIndex):
                d = df.index.to_pydatetime()
            else:
                d = df.index.values
        else:
            d = df[self.name].values

        # d = np_get_by_col_name(self.name, data, column_names)

        return TransformDelayed(d, self.transforms, self.name)


class TransformDelayed:
    def __init__(self, data, transforms=[], data_name=""):
        self.data = data
        self.data_name = data_name
        self.transforms = transforms

    def eval(self):
        v = self.data
        for tr in self.transforms:
            v = tr.apply(v)
        return v

    def get_name(self):
        res = [self.data_name]
        for tr in self.transforms:
            res.append(tr.get_name())

        return ".".join(res)


def kwargs_to_str(kwargs, brackets=True):
    if kwargs:
        params = [f"{k}={v}" for k, v in kwargs.items() if type(v) not in [np.ndarray, list]]
        if params:
            res = "%s" % ", ".join(params)
            return res if not brackets else f"({res})"

    return "()"


df_in_process_cache = None


def pre_cache(np_index, np_data, pd_columns):
    global df_in_process_cache
    if df_in_process_cache is None:
        df_in_process_cache = pd.DataFrame(np_data, columns=pd_columns)
        df_in_process_cache['index'] = np_index
        df_in_process_cache.set_index("index", inplace=True)
        return True
    return False


def pre_cache_hdf(hdf_path):
    global df_in_process_cache
    if df_in_process_cache is None:
        df_in_process_cache = pd.read_hdf(hdf_path, key="table")
        return True
    return False


def np_get_by_col_name(col_name, np_data, pd_columns):
    try:
        col_idx = pd_columns.index(col_name)
    except ValueError:
        raise ValueError(f"Column name `{col_name}` not present in dataset")

    if col_idx == -1:
        raise ValueError(f"Column name `{col_name}` not present in dataset")

    return np_data[:, col_idx]


def fn_call_wrapper(df, hdf_path, fn, kwargs, transforms):
    if df is None:
        print("-> pre_cache_hdf")
        pre_cache_hdf(hdf_path)
        global df_in_process_cache
        df = df_in_process_cache

    input_names = dict()
    for kw_name, v in kwargs.items():
        if type(v) == Col:
            kwargs[kw_name] = kwargs[kw_name].get_value(df)
            input_names[kw_name] = kwargs[kw_name].get_name()

    for k, v in kwargs.items():
        if type(v) == TransformDelayed:
            kwargs[k] = v.eval()

    return [TransformDelayed(fn(**kwargs), transforms).eval(), input_names]


def calc_all_config(data: pd.DataFrame, config: Dict, n_jobs=1,
                    dask=False, verbose=False, max_nbytes=None,
                    dump_folder=None) -> pd.DataFrame:
    """
    Example usage:

    ```
    config = dict(
        # prefix aka namespace
        prefix_1=[
            # config for single feature function grid
            dict(
                # feature function callable
                fn=features.datetime.holidays.f1,
                # parameter grid
                pg=km(country_name=['US', 'RU']),
                # data mapping
                dm=km(timestamps=[
                    # can ge referenced by name, "index" is reserved name to refer pandas `data.index`
                    "index",
                    # km.col - helper to wrap column name with "pre" transforms
                    # t.symlog, t.custom_transform, etc - wrapper to highlight available transformations
                    km.col("index").t([t.custom_transform(1, "OK")]),
                    # km.col("close").t([t.symlog()])
                ]),
                # post transforms
                tr=[t.symlog()]
            )
        ]
    )
    df1 = calc_all_config(df, config=config) # n_jobs==1 by default // single thread
    ```

    :param data: pandas DataFrame
    :param config:
    :param n_jobs: joblib n_jobs parameter parallel process count,
                   1 = don't use paralell code, -1 == CPU count
    :return:
    """

    if dump_folder is not None and dask:
        raise AttributeError("`dask` mode can be used with remote cluster, `dump_folder` should be None`")

    joblib_client = "loky"
    if dask:
        joblib_client = "dask"
        # disable max_nbytes while in dask (can be remote cluster)
        max_nbytes = None

    print("joblib_client", joblib_client)
    with joblib.parallel_backend(joblib_client, n_jobs=n_jobs):
        fn_parallel_list = []
        for prefix, fn_settings_list in config.items():
            for fn_settings in fn_settings_list:
                fn = fn_settings['fn']
                pg = fn_settings['pg']
                dm = fn_settings['dm']
                transforms_post = fn_settings['tr']

                kwargs_list = list(ParameterGrid({**pg, **dm}))
                for kwargs in kwargs_list:
                    kwargs_dump = dict(
                        inputs=dict(),
                        params=dict()
                    )
                    for kw_name in list(kwargs.keys()):
                        if kw_name in pg:
                            kwargs_dump['params'][kw_name] = kwargs[kw_name]
                        if kw_name in dm:
                            if type(kwargs[kw_name]) == str:
                                # automatic "col_name" -> km.col("col_name")
                                kwargs[kw_name] = km.col(kwargs[kw_name])

                            # passed to fn_call_wrapper
                            # kwargs[kw_name] = kwargs[kw_name].get_value(data)
                            # kwargs_dump['inputs'][kw_name] = kwargs[kw_name].get_name()

                    fn_parallel_list.append([fn, kwargs, transforms_post, dict(prefix=prefix, kwargs=kwargs_dump)])

        with joblib.Parallel(
                max_nbytes=max_nbytes,
                n_jobs=n_jobs,
                # prefer="threads" # usually it's a bad idea, but may be useful for future
        ) as parallel:
            # cache data on workers

            # print(parallel.n_jobs, parallel.)
            # tasks_count = n_jobs if n_jobs != -1 else joblib.cpu_count()
            # cache_res = parallel(
            #     [joblib.delayed(pre_cache)(data.index, data.values, data.columns) for i in range(tasks_count)]
            # )
            #
            # print("cache_res", cache_res)

            # np_columns = list(data.columns)
            # np_data = data.values
            # np_index = data.index
            # if isinstance(np_index, pd.DatetimeIndex):
            #     np_index = np_index.to_pydatetime()
            # # append index as last column
            # # print(np_data[:, :-1])
            # # print(np_index)
            # np_data = np.c_[np_index, np_data]
            # np_columns = ['index'] + np_columns

            data_file_name = None
            data_file_path = None
            if dump_folder:
                data_file_name = uuid.uuid4()
                data_file_path = f"{dump_folder}/{data_file_name}"
                data.to_hdf(data_file_path, key="table")

            try:
                df_data = None
                if not data_file_path:
                    df_data = data

                results = parallel(
                    joblib.delayed(fn_call_wrapper)(df_data, data_file_path, func, kwargs, transforms) for [func, kwargs, transforms, _] in
                    fn_parallel_list
                )

                result_dict = dict()
                i = 0
                for [fn, _, transforms_post, dump] in fn_parallel_list:
                    prefix = dump['prefix']
                    kwargs_dump = dump['kwargs']
                    transforms_post_str = ".".join([t.get_name() for t in transforms_post])

                    fn_name = fn.__name__
                    kwargs_str = [kwargs_to_str(kwargs_dump['inputs'], brackets=False)]
                    if kwargs_dump['params']:
                        kwargs_str.append(kwargs_to_str(kwargs_dump['params'], brackets=False))

                    kwargs_str = ", ".join(kwargs_str)
                    result_name = f"{prefix}.{fn_name}({kwargs_str})"
                    if transforms_post_str:
                        result_name += f".{transforms_post_str}"

                    # TODO: @multiple_outputs_task if results[i] is not 1D np.array we can split it to
                    #       multiple outputs with `_{i}` postfix

                    result_dict[result_name] = results[i]
                    i += 1

            finally:
                if data_file_path:
                    os.remove(data_file_path)

            return pd.DataFrame(result_dict)


class KeyMap:
    col = Col

    @staticmethod
    def __call__(**kwargs):
        return kwargs


km = KeyMap()
t = Transforms
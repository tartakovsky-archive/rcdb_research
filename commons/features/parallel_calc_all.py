import os
import sys
import json
import uuid
import copy
import flock
import joblib
import pandas as pd
import numpy as np
import subprocess


from typing import List, Dict, Callable
from sklearn.model_selection import ParameterGrid
from commons.utils import np_to_file, np_from_file, kwargs_to_str, json_from_file, \
    json_to_folder, chunks, FnSerializer, AttrDict, generate_constraints_function
from commons.features.transformations import TransformObj, Transforms, TransformsMixin, TransformDelayed


class Column(TransformsMixin):
    def __init__(self, name, transforms=None):
        self.name = name
        self.transforms = transforms

        if self.transforms is None:
            self.transforms = []

        self.value_delayed = None

    def t(self, transforms: List[TransformObj]):
        self.transforms += transforms
        return self

    def get_value(self, df):
        if self.value_delayed is None:
            self.value_delayed = TransformDelayed(df[self.name], self.transforms, self.name)

        return self.value_delayed

    def get_name(self):
        res = [self.name]
        for tr in self.transforms:
            res.append(tr.get_name())

        return ".".join(res)

    def eval(self, df):
        return self.get_value(df).eval()

    def to_dict(self):
        return dict(
            type='Column',
            name=self.name,
            transforms=[t.to_dict() for t in self.transforms]
        )

    def __repr__(self):
        return self.get_name()

    def __str__(self):
        return self.get_name()

    @classmethod
    def from_dict(cls, d):
        return cls(d['name'], [TransformObj.from_dict(t) for t in d['transforms']])


class KeyMap:
    col = Column

    @staticmethod
    def __call__(**kwargs):
        return kwargs


km = KeyMap()
t = Transforms
col = Column


class NpDataFrame:
    def __init__(self, np_array, columns):
        self.np_array = np_array
        self.columns = columns

    def __getitem__(self, item):
        if type(item) == list:
            return self.__get_multiple(item)
        else:
            return self.__get_single(item)

    def __get_multiple(self, columns: List[str]) -> np.ndarray:
        return np.c_[[
            self.__np_get_by_col_name(column, self.np_array, self.columns)
            for column in columns
        ]]

    def __get_single(self, column: str) -> np.ndarray:
        return self.__np_get_by_col_name(column, self.np_array, self.columns)

    @staticmethod
    def __np_get_by_col_name(col_name, np_data, pd_columns):
        try:
            col_idx = pd_columns.index(col_name)
        except ValueError:
            raise ValueError(f"Column name `{col_name}` not present in dataset")

        if col_idx == -1:
            raise ValueError(f"Column name `{col_name}` not present in dataset")

        return np_data[:, col_idx]


class FnTask:
    name_to_class_dict = {
        "Column": Column,
        "TransformObj": TransformObj
    }

    def __init__(self, task_meta: Dict):
        self.task_meta = copy.deepcopy(task_meta)
        self.fn = FnSerializer.get_by_name(task_meta['fn_name'])
        self.alias = self.fn.__name__
        if 'alias' in task_meta:
            self.alias = task_meta['alias']
        self.prefix = task_meta['prefix']
        self.input_names = {}
        self.params = task_meta['params']
        self.transform_post = [TransformObj.from_dict(t) for t in task_meta['transform_post']]
        self.kwargs = task_meta['kwargs']

        self.de_serialize()

    def eval(self, df):
        for k in self.kwargs:
            if hasattr(self.kwargs[k], 'eval'):
                self.kwargs[k] = self.kwargs[k].eval(df)
        return self

    def calc(self):
        return TransformDelayed(self.fn(**self.kwargs), self.transform_post).eval()

    def get_name(self):
        transforms_post_str = ".".join([t.get_name() for t in self.transform_post])
        fn_name = self.alias

        kwargs_str = [kwargs_to_str(self.input_names, brackets=False)]
        if self.params:
            kwargs_str.append(kwargs_to_str(self.params, brackets=False))

        kwargs_str = ", ".join(kwargs_str)
        result_name = f'{self.prefix}.{fn_name}({kwargs_str})'
        if transforms_post_str:
            result_name += "." + transforms_post_str
        return result_name

    def serialize(self, as_dict=False):
        if as_dict:
            return json.dumps(self.task_meta)
        else:
            return self.get_name()

    def de_serialize(self):
        for kw_name in self.kwargs.keys():
            if type(self.kwargs[kw_name]) == dict and 'type' in self.kwargs[kw_name]:
                self.kwargs[kw_name] = self.name_to_class_dict[
                    self.kwargs[kw_name]['type']
                ].from_dict(
                    self.kwargs[kw_name]
                )

                if type(self.kwargs[kw_name]) == Column:
                    self.input_names[kw_name] = self.kwargs[kw_name].get_name()


class JobTaskWrapper:
    def __init__(self, job_meta, data_cache: Dict = None, output_callback: Callable = None):
        self.data_cache = {}
        if data_cache is not None:
            self.data_cache = data_cache
        self.job_meta = job_meta

        if self.job_meta['n_jobs'] == 1 and self.job_meta['output_folder'] is None:
            self.output_callback = self.__sequential_jobs_feature_output_callback
        else:
            self.output_callback = self.__concurrent_jobs_feature_output_callback

    def __call__(self, tasks):
        return self.process_tasks(self.__fetch_tasks(tasks))

    @staticmethod
    def __fetch_tasks(task_list):
        if type(task_list) == str:
            return json_from_file(task_list)
        else:
            return task_list

    def process_tasks(self, tasks):
        df = self.get_data()

        results = []
        for task_meta in tasks:
            t = FnTask(task_meta)
            feature_output = t.eval(df).calc()

            if not self.job_meta['benchmark']:
                feature_output_file_name = self.output_callback(
                    self.job_meta,
                    t,
                    feature_output
                )
                results.append(feature_output_file_name)

        return results

    @staticmethod
    def __concurrent_jobs_feature_output_callback(job_meta, task: FnTask, fn_output: np.ndarray):
        feature_output_file = str(uuid.uuid4().hex)
        task_serialized = task.serialize(as_dict=False)
        if job_meta['name_as_dict']:
            with open(os.path.join(job_meta['output_folder'], "feature_info", feature_output_file), "w") as f:
                json.dump({**task.task_meta, '_name': task_serialized}, f)

        feature_output_path = os.path.join(job_meta['output_folder'], feature_output_file)
        pd.Series(fn_output, name=task_serialized).to_pickle(feature_output_path)
        return feature_output_file

    @staticmethod
    def __sequential_jobs_feature_output_callback(job_meta, task: FnTask, fn_output: np.ndarray):
        return [task.serialize(as_dict=False), fn_output]

    def get_data(self):
        is_file = type(self.job_meta['input_data']) == str
        if is_file:
            fname = self.job_meta['input_data']
        else:
            fname = ""

        if fname not in self.data_cache:
            if is_file:
                ndarr = np_from_file(fname)
            else:
                ndarr = self.job_meta['input_data']

            if len(ndarr.shape) == 1:
                ndarr = ndarr.reshape(-1, 1)

            self.data_cache[fname] = NpDataFrame(ndarr, self.job_meta['data_columns'])

        return self.data_cache[fname]


class JobHandler:
    def __init__(self, job_meta: Dict, proc_num=None):
        self.task_is_dict = True
        self.job_meta = job_meta

        self.task_list = self.__get_task_list()
        self.task_list_current_position = 0

        self.debug = self.job_meta['debug']
        self.proc_num = proc_num

        if self.proc_num is None:
            self.proc_num = str(uuid.uuid4())

    @classmethod
    def create_from_argv(cls):
        return cls(
            job_meta=json.loads(sys.argv[1]),
            proc_num=sys.argv[2]
        )

    def __get_task_list(self):
        if 'task_folder' in self.job_meta:
            if type(self.job_meta['task_folder']) == str:
                if not os.path.isdir(self.job_meta['task_folder']):
                    raise AttributeError(f"job_meta['task_folder'] `{self.job_meta['task_folder']}` doesn't exists!")

                return os.listdir(self.job_meta["task_folder"])
        elif 'task_list' in self.job_meta:
            return self.job_meta['task_list']

    def run_job(self):
        jw = JobTaskWrapper(self.job_meta)
        results = []
        task_counter = 0
        task_list_len = self.get_task_count()
        while self.task_list_current_position < task_list_len:
            t = self.get_next_task()
            if self.debug:
                print(f"{self.proc_num} >> task_list_current_postion: {self.task_list_current_position}")
            if t:
                results += jw.process_tasks(t)
                task_counter += 1

                if self.debug:
                    print(f"{self.proc_num} >> {task_counter}")

        return results

    def get_next_task(self):
        if not self.task_list:
            return []

        if self.is_file_tasks():
            task_file = self.task_list[self.task_list_current_position]
            with open(os.path.join(self.job_meta['job_folder'], 'job.lock'), 'w') as fp:
                with flock.Flock(fp, flock.LOCK_EX) as lock:  # noqa
                    # exclusive lock is acquired here
                    task_file = os.path.join(self.job_meta['task_folder'], task_file)
                    try:
                        task = json.load(open(task_file, "r"))
                        os.remove(task_file)
                        return task
                    except Exception:
                        return None
                    finally:
                        self.task_list_current_position += 1
        else:
            self.task_list_current_position += 1
            return self.task_list

    def get_task_count(self):
        if self.is_file_tasks():
            return len(self.task_list)

        return 1

    def is_file_tasks(self):
        return 'job_folder' in self.job_meta and 'task_folder' in self.job_meta


class JobOutput:
    def __init__(self, output_folder=None, results=None, index=None, stdout=None, stderr=None):
        if output_folder is None and results is None:
            raise AttributeError("Both `output_folder` and `results` can't be None")

        self.output_folder = output_folder
        self.results = results
        self.stdout = stdout
        self.stderr = stderr
        self.index = index

    def get_pandas(self):
        return self.__build_pandas_dataframe(self.get_results())

    def get_results(self):
        if self.results is not None:
            return self.__list_generator(self.results)

        return self.__load_output_folder_data(self.output_folder)

    def get_info(self):
        if self.output_folder is not None:
            for item in self.__load_info_folder_data(self.output_folder):
                yield item

    def count_results(self):
        if self.results is not None:
            return len(self.results)

        return len(os.listdir(self.output_folder + "/")) - 1  # minus feature_info folder

    def __build_pandas_dataframe(self, results):
        df = pd.DataFrame({name: v for name, v in results})
        if self.index is not None:
            df.index = self.index
        return df

    @staticmethod
    def __list_generator(l):
        for item in l:
            yield item

    @staticmethod
    def __load_output_folder_data(output_folder):
        files = os.listdir(output_folder)
        for fname in files:
            if fname.endswith('feature_info'):
                continue

            series = pd.read_pickle(os.path.join(output_folder, fname))
            yield series.name, series

    @staticmethod
    def __load_info_folder_data(output_folder):
        file_folder = os.path.join(output_folder, 'feature_info')
        files = os.listdir(file_folder)
        for fname in files:
            with open(os.path.join(file_folder, fname), 'r') as f:
                data = json.load(f)
                yield data.pop('_name'), data


class JobManager:
    def __init__(self, data: pd.DataFrame, config: Dict,
                 n_jobs=1, batch_size=200, benchmark=False, temp_folder=None,
                 python_executable=None, debug=False,
                 add_feature_ids=False, name_as_dict=False
                 ):
        if n_jobs != 1 and temp_folder is None:
            raise AttributeError("temp_folder is required for parallel_execution")

        if n_jobs == -1:
            n_jobs = joblib.cpu_count()

        if data.index.tz is not None:
            raise ValueError("Timezones not allowed in the input DataFrame. "
                             "Use `df.index = pd.to_datetime(df.index).tz_localize(None)` to remove tz info.")

        self.job_meta = self.create_job(
            data,
            config=config,
            n_jobs=n_jobs,
            temp_folder=temp_folder,
            batch_size=batch_size,
            benchmark=benchmark,
            add_feature_ids=add_feature_ids
        )
        self.job_meta['debug'] = debug
        self.job_meta['name_as_dict'] = name_as_dict
        self.python_executable = python_executable

    def run_job(self):
        if self.job_meta['n_jobs'] == 1:
            if 'output_folder' in self.job_meta and self.job_meta['output_folder'] is not None:
                JobHandler(self.job_meta).run_job()
                return JobOutput(
                    output_folder=self.job_meta['output_folder'],
                    index=self.job_meta['index_data']
                )
            else:
                return JobOutput(
                    results=JobHandler(self.job_meta).run_job(),
                    index=self.job_meta['index_data']
                )
        else:
            python_path = self.python_executable if self.python_executable is not None else sys.executable
            arg_str = json.dumps(self.job_meta)

            command = f"{python_path} -c 'from commons.features.parallel_calc_all import" \
                      f" JobHandler;job_handler = JobHandler.create_from_argv();job_handler.run_job()' '{arg_str}'"
            command_list = []
            for i in range(self.job_meta['n_jobs']):
                command_tmp = command + f" {i} &"
                command_list.append(command_tmp)

            proc = subprocess.Popen(
                " ".join(command_list + ['wait']),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                shell=True, encoding="UTF-8"
            )
            stdout = proc.stdout.read()

            data_index = pd.DatetimeIndex(
                np_from_file(self.job_meta['index_data'])
            )
            return JobOutput(output_folder=self.job_meta['output_folder'],
                             index=data_index,
                             stdout=stdout, stderr=proc.stderr.read())

    @staticmethod
    def __init_folders(temp_folder):
        task_id = f"{uuid.uuid4()}"
        job_folder = os.path.join(temp_folder, task_id)

        data_file_name = "input.data"
        data_file_path = os.path.join(job_folder, data_file_name)

        output_folder = os.path.join(job_folder, 'out')
        feature_info = os.path.join(output_folder, 'feature_info')
        task_folder = os.path.join(job_folder, 'tasks')

        os.mkdir(job_folder)
        os.mkdir(output_folder)
        os.mkdir(feature_info)
        os.mkdir(task_folder)

        return data_file_path, task_folder, output_folder, job_folder

    def create_job(
            self, data: pd.DataFrame, config: Dict, n_jobs=1, temp_folder=None, batch_size=1000,
            verbose=0, benchmark=False, add_feature_ids=False
    ):
        output_folder = None
        if n_jobs != 1 or temp_folder is not None:
            (
                data_file_path,
                task_folder,
                output_folder,
                job_folder
            ) = self.__init_folders(temp_folder)

        fn_parallel_list = []
        inputs_set = set()

        for prefix, fn_settings_list in config.items():
            for fn_settings in fn_settings_list:
                fn_name = FnSerializer.get_full_name(fn_settings['fn'])
                pg = fn_settings.get('pg', km())
                dm = fn_settings['dm']
                cn = fn_settings.pop('cn', None)

                transforms_post = []
                if 'tr' in fn_settings and fn_settings['tr'] is not None:
                    transforms_post = [t.to_dict() for t in fn_settings['tr']]

                kwargs_list = list(ParameterGrid({**pg, **dm}))
                constraint = generate_constraints_function(cn) if cn else None

                feature_num = 0
                for kwargs in kwargs_list:
                    input_params = dict()
                    for kw_name in list(kwargs.keys()):
                        if kw_name in pg:
                            input_params[kw_name] = kwargs[kw_name]
                        if kw_name in dm:
                            if type(kwargs[kw_name]) == str:
                                # automatic "col_name" -> km.col("col_name")
                                kwargs[kw_name] = km.col(kwargs[kw_name])

                            inputs_set.add(kwargs[kw_name].name)
                            kwargs[kw_name] = kwargs[kw_name].to_dict()

                    if constraint is not None and not constraint(AttrDict(input_params)):
                        continue

                    feature_data = dict(
                        fn_name=fn_name,
                        prefix=prefix,
                        params=input_params,
                        transform_post=transforms_post,
                        kwargs=kwargs
                    )
                    if 'alias' in fn_settings and fn_settings['alias'] is not None:
                        feature_data['alias'] = fn_settings['alias']

                    if add_feature_ids:
                        if 'alias' not in feature_data:
                            feature_data['alias'] = fn_settings['fn'] if type(fn_settings['fn']) == str\
                                else fn_settings['fn'].__name__
                        feature_data['alias'] = f"{feature_num}-{feature_data['alias']}"

                    feature_num += 1
                    fn_parallel_list.append(feature_data)

        if n_jobs != 1:
            # dump task files
            task_files = [json_to_folder(chnk, task_folder) for chnk in chunks(fn_parallel_list, batch_size)]

        if verbose == 1:
            print(len(task_files))

        if "index" in inputs_set:
            data_index = data.index
            if isinstance(data_index, pd.DatetimeIndex):
                data_index = data_index.to_pydatetime()
            data['index'] = data_index

        if n_jobs == 1:
            return dict(
                n_jobs=n_jobs,
                task_list=fn_parallel_list,
                input_data=data[list(inputs_set)].values,
                index_data=data.index,
                data_columns=list(inputs_set),
                benchmark=benchmark,
                output_folder=output_folder,
            )
        else:
            __data_file_path = np_to_file(data_file_path, data[list(inputs_set)].values)
            __index_file_path = np_to_file(f'{data_file_path}.index', data.index.values)
            return dict(
                n_jobs=n_jobs,
                job_folder=job_folder,
                input_data=__data_file_path,
                index_data=__index_file_path,
                output_folder=output_folder,
                task_folder=task_folder,
                data_columns=list(inputs_set),
                benchmark=benchmark
            )

        # finally:
        #     if is_tmp_folder:
        #         os.remove(temp_folder)
        #     else:
        #         try:
        #             os.remove(job_folder)
        #             os.remove(task_folder)
        #             os.remove(data_file_path)
        #         except Exception:
        #             raise

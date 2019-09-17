from hyperopt import space_eval
import pandas as pd
import json_tricks as jsont
import warnings

class HOTrial:
    """
    Class for representing one trial gathered from hyperopt.fmin(...)
    """
    ############
    # Initialization
    ############
    def __init__(self, tid, params, metrics):
        self._tid = tid
        self._params = params
        self._metrics = metrics

    @classmethod
    def from_trial(cls, trial, space, drop_loss_status=False):
        tid = trial['tid']

        param_idxs = {
            k: v[0] if len(v) > 0 else None
            for k, v in trial['misc']['vals'].items()
        }
        params = space_eval(space, param_idxs)

        metrics = trial['result'].copy()

        if drop_loss_status:
            metrics.pop('loss', None)
            metrics.pop('status', None)

        return cls(tid, params, metrics)

    @classmethod
    def from_dict(cls, d):
        tid = d.get('tid', 0)
        params = d.get('params', dict())
        metrics = d.get('metrics', dict())
        return cls(tid, params, metrics)

    def to_dict(self):
        return dict(
            tid=self._tid,
            params=self._params,
            metrics=self._metrics,
        )

    ############
    # Public methods
    ############
    @property
    def tid(self):
        return self._tid

    @property
    def params(self):
        return self._params

    @property
    def metrics(self):
        return self._metrics

    @property
    def result(self):
        return {**self._params, **self._metrics}

    @property
    def params_df(self):
        return pd.DataFrame({**self._params}, index=[self._tid])

    @property
    def metrics_df(self):
        return pd.DataFrame({**self._metrics}, index=[self._tid])

    @property
    def result_df(self):
        return pd.DataFrame({**self.result}, index=[self._tid])



class HOExperiment:
    """
    Class for representing a set of trials gathered from hyperopt.fmin(...)
    """
    ############
    # Initialization
    ############
    def __init__(self, trials):
        self.trials = trials

    @classmethod
    def from_hyperopt_trials(cls, hyperopt_trials, space, drop_loss_status=False):
        trials = [HOTrial.from_trial(t, space, drop_loss_status) for t in hyperopt_trials.trials]
        return cls(trials)

    @classmethod
    def from_dict(cls, list_of_dicts):
        trials = [HOTrial.from_dict(d) for d in list_of_dicts]
        return cls(trials)

    def to_dict(self):
        return [t.to_dict() for t in self.trials]

    @classmethod
    def from_json_file(cls, path):
        with open(path, 'r') as f:
            json = jsont.load(f)

        return cls.from_dict(json)

    def to_json_file(self, path):
        warnings.simplefilter("ignore")
        with open(path, 'w') as f:
            res = jsont.dump(self.to_dict(), f)
        warnings.simplefilter("default")
        return res

    ############
    # Public methods
    ############

    ######
    # Dicts
    ######
    @property
    def ids(self):
        return [t.tid for t in self.trials]

    @property
    def metrics(self):
        return [t.metrics for t in self.trials]

    @property
    def params(self):
        return [t.params for t in self.trials]

    @property
    def results(self):
        return [t.result for t in self.trials]


    ######
    # Pandas Dataframes
    ######
    @property
    def metrics_df(self):
        return pd.DataFrame(self.metrics, index=self.ids)

    @property
    def params_df(self):
        return pd.DataFrame(self.params, index=self.ids)

    @property
    def results_df(self):
        return pd.concat([self.params_df, self.metrics_df], axis=1, sort=False)


class HOAnalysis:
    """
    Class for analysing HOExperiment contents
    """
    ############
    # Initialization
    ############
    def __init__(self, experiment):
        self.experiment = experiment

    ############
    # Public methods
    ############
    def n_best(self, up_to_n=5, recall_threshold=0.05, precision_quantile=0.9):
        best = self.experiment.results_df
        best = best[best['recall'] >= recall_threshold]
        best = best[best['precision'] >= best['precision'].quantile(precision_quantile)]
        best = best.sort_values(by=['recall'], ascending=False)
        best = best.head(up_to_n)

        best_ids = best.index.tolist()

        indexed_trials = dict(map((lambda t: (t.tid, t)), self.experiment.trials)) 
        best_trials = [indexed_trials[idx] for idx in best_ids]

        return HOExperiment(best_trials)

import pytest
import pandas as pd

from rcdb_research.hyperopt_ext import HOTrial, HOExperiment, HOAnalysis


def test_HOTrial_init(mocker):
    tid = mocker.Mock()
    params = mocker.Mock()
    metrics = mocker.Mock()

    ho_trial = HOTrial(tid, params, metrics)

    assert ho_trial.tid is tid
    assert ho_trial.params is params
    assert ho_trial.metrics is metrics


def test_HOTrial_from_trial(mocker):
    trial = {
        'tid': 1,
        'result': {},
        'misc': {
            'vals': {}
        }
    }
    mock = mocker.patch('rcdb_research.hyperopt_ext.space_eval', autospec=True)
    assert isinstance(HOTrial.from_trial(trial, []), HOTrial)
    mock.assert_called_once()


def test_HOTrial_from_dict():
    assert isinstance(HOTrial.from_dict({}), HOTrial)


def test_HOTrial_to_dict():
    assert isinstance(HOTrial({}, {}, {}).to_dict(), dict)


@pytest.mark.parametrize(
    'prop',
    ['result', 'params_df', 'metrics_df', 'result_df']
)
def test_HOTrial_additional_propetries(prop):
    assert getattr(HOTrial({}, {}, {}), prop) is not None


def test_HOExperiment_init(mocker):
    trials = mocker.Mock()
    assert HOExperiment(trials).trials is trials


@pytest.fixture
def ho_experiment():
    return HOExperiment.from_dict([{}])


def test_HOExperiment_from_dict():
    assert isinstance(HOExperiment.from_dict([{}]), HOExperiment)


def test_HOExperiment_to_dict(ho_experiment):
    assert isinstance(ho_experiment.to_dict(), list)


def test_HOExperiment_to_json_file(ho_experiment, tmp_path):
    res_path = tmp_path / 'res.json'
    assert isinstance(ho_experiment.to_json_file(res_path), str)
    assert res_path.read_text()


def test_HOExperiment_from_json_file(tmp_path):
    source_file = tmp_path / 'data.json'
    source_file.write_text('[{}]')

    assert isinstance(HOExperiment.from_json_file(source_file), HOExperiment)


@pytest.mark.parametrize(
    'prop',
    ['ids', 'metrics', 'params', 'results', 'metrics_df', 'params_df', 'results_df']
)
def test_HOExperiment_additional_propetries(ho_experiment, prop):
    assert getattr(ho_experiment, prop) is not None


def test_HOAnalysis_init(mocker):
    experiment = mocker.Mock()
    assert HOAnalysis(experiment).experiment is experiment


def test_HOAnalysis_n_best(ho_experiment, mocker):
    mock = mocker.patch(f'{__name__}.HOExperiment.results_df', new_callable=mocker.PropertyMock)
    mock.return_value = pd.DataFrame({'recall': [], 'precision': []})
    assert isinstance(HOAnalysis(ho_experiment).n_best(), HOExperiment)

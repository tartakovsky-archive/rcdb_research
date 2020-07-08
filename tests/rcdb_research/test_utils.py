import pytest
import numpy as np

from rcdb_research.utils import split_dict_array_values


@pytest.mark.parametrize(
    'params, test_res',
    [
        (
            dict(splits=2),
            [
                dict(
                    a=np.array([1, 2, 3, 4, 5]),
                    b=np.array([-1, -2, -3, -4, -5])
                ),
                dict(
                    a=np.array([6, 7, 8, 9, 10]),
                    b=np.array([-6, -7, -8, -9, -10])
                ),
            ]
        ),

        (
            dict(splits=3),
            [
                dict(
                    a=np.array([1, 2, 3, 4]),
                    b=np.array([-1, -2, -3, -4])
                ),
                dict(
                    a=np.array([5, 6, 7]),
                    b=np.array([-5, -6, -7])
                ),
                dict(
                    a=np.array([8, 9, 10]),
                    b=np.array([-8, -9, -10])
                ),

            ],
        ),

        (
            dict(split_sizes=[1, 5, 4]),
            [
                dict(
                    a=np.array([1]),
                    b=np.array([-1])
                ),
                dict(
                    a=np.array([2, 3, 4, 5, 6]),
                    b=np.array([-2, -3, -4, -5, -6])
                ),
                dict(
                    a=np.array([7, 8, 9, 10]),
                    b=np.array([-7, -8, -9, -10])
                ),
            ]
        ),

        (
            dict(split_sizes=[0, 1, 2, 2, 0, 5, 0, 0]),
            [
                dict(
                    a=np.array([]),
                    b=np.array([])
                ),
                dict(
                    a=np.array([1]),
                    b=np.array([-1])
                ),
                dict(
                    a=np.array([2, 3]),
                    b=np.array([-2, -3])
                ),
                dict(
                    a=np.array([4, 5]),
                    b=np.array([-4, -5])
                ),
                dict(
                    a=np.array([]),
                    b=np.array([])
                ),
                dict(
                    a=np.array([6, 7, 8, 9, 10]),
                    b=np.array([-6, -7, -8, -9, -10])
                ),
                dict(
                    a=np.array([]),
                    b=np.array([])
                ),
                dict(
                    a=np.array([]),
                    b=np.array([])
                ),
            ]
        )
    ]
)
def test_split_dict_array_values(params, test_res):
    d = dict(
        a=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        b=np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
    )
    res = split_dict_array_values(d, **params)
    assert len(res) == len(test_res)

    for r, t in zip(res, test_res):
        assert r.keys() == t.keys()
        for k in r:
            assert np.array_equal(
                r[k], t[k]
            )

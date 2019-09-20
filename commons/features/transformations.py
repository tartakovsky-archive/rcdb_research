import numba # noqa
import numpy as np
from commons.utils import kwargs_to_str


@numba.jit
def symlog2_(x):
    C = 0  # parameter
    return np.sign(x) * (np.log2(1 + np.abs(x) / (10 ** C)))


@numba.jit
def power(x, p):
    return np.sign(x) * np.abs(x)**p


@numba.jit
def root(x, p):
    return power(x, 1/p)


@numba.jit
def symlog_symlog(x):
    return symlog2_(symlog2_(x))


@numba.jit
def clip(inp, crop_perc):
    X = inp
    lp, hp = np.percentile(X, [crop_perc, 100-crop_perc])

    X[X > hp] = hp
    X[X < lp] = lp

    return X


def none(x):
    return x


name_to_fn = dict(
    symlog=symlog2_,
    power=power,
    root=root,
    symlog_symlog=symlog_symlog,
    clip=clip,
    none=none
)


class TransformObj:
    def __init__(self, fn_name: str, **kwargs):
        self.fn_name = fn_name
        self.__fn = name_to_fn[fn_name]

        if not kwargs:
            kwargs = {}
        self.kwargs = kwargs

    def apply(self, data):
        return self.__fn(data, **self.kwargs)

    def get_name(self):
        return f"{self.fn_name}{kwargs_to_str(self.kwargs)}"

    def __repr__(self):
        return self.get_name()

    def __str__(self):
        return self.get_name()

    def to_dict(self):
        return dict(
            type='TransformObj',
            fn_name=self.fn_name,
            kwargs=self.kwargs
        )

    @classmethod
    def from_dict(cls, d):
        return cls(d['fn_name'], **d['kwargs'])


class Transforms:
    @staticmethod
    def none():
        return TransformObj("none")

    @staticmethod
    def symlog():
        return TransformObj("symlog")

    @staticmethod
    def symroot2():
        return TransformObj("root", p=2)

    @staticmethod
    def symroot3():
        return TransformObj("root", p=3)

    @staticmethod
    def sympower2():
        return TransformObj("power", p=2)

    @staticmethod
    def sympower3():
        return TransformObj("power", p=3)

    @staticmethod
    def symlog_symlog():
        return TransformObj("symlog_symlog")

    @staticmethod
    def clip():
        return TransformObj("clip", crop_perc=5)


class TransformsMixin:
    def none(cls):
        cls.t([Transforms.none()])
        return cls

    def symlog(self):
        self.t([Transforms.symlog()])
        return self

    def symroot2(self):
        self.t([TransformObj("root", p=2)])
        return self

    def symroot3(self):
        self.t([TransformObj("root", p=3)])
        return self

    def sympower2(self):
        self.t([TransformObj("power", p=2)])
        return self

    def sympower3(self):
        self.t([TransformObj("power", p=3)])
        return self

    def symlog_symlog(self):
        self.t([TransformObj("symlog_symlog")])
        return self

    def clip(self):
        self.t([TransformObj("clip", crop_perc=5)])
        return self


class TransformDelayed:
    def __init__(self, data, transforms=tuple(), data_name=""):
        self.data = data
        self.data_name = data_name
        self.transforms = transforms

    def eval(self):
        v = self.data
        for tr in self.transforms:
            v = tr.apply(v)
        return v

import os
import pip
from setuptools import setup
from pip._internal.req import parse_requirements
from pip._internal.download import PipSession

REQUIREMENTS_DIR = "requirements"

pip_session = PipSession()


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


def parse_reqs(path):
    return [str(r.req) for r in parse_requirements(path, session=pip_session)]


INSTALL_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.txt"))
DEV_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.dev.txt"))


def get_numpy():
    for reqs in [INSTALL_REQUIREMENTS, DEV_REQUIREMENTS]:
        for i, req in enumerate(reqs):
            if req.startswith("numpy"):
                return reqs.pop(i)


numpy = get_numpy()
if not numpy:
    raise ValueError("No numpy in requirements")

print(f"Installing {numpy}")

install(numpy)
setup(
    name='commons',
    packages=['commons'],
    install_requires=INSTALL_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS
    },
)

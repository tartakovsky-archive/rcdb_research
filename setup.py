import os
import pip
from setuptools import setup, find_packages
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
SETUP_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.pre.txt"))


for build_req in SETUP_REQUIREMENTS:
    print(f'{build_req} installation...')
    install(build_req)


setup(
    name='commons',
    packages=find_packages(include=["commons*"]),
    install_requires=INSTALL_REQUIREMENTS + SETUP_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS
    },
)

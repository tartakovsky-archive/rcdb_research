import os
from setuptools import setup
from pip._internal.req import parse_requirements
from pip._internal.download import PipSession

REQUIREMENTS_DIR = "requirements"

pip_session = PipSession()
INSTALL_REQUIREMENTS = list(parse_requirements(os.path.join(REQUIREMENTS_DIR, "requirements.txt"), session=pip_session))
DEV_REQUIREMENTS = list(parse_requirements(os.path.join(REQUIREMENTS_DIR, "requirements.dev.txt"), session=pip_session))
INSTALL_REQUIREMENTS = [str(req.req) for req in INSTALL_REQUIREMENTS]
DEV_REQUIREMENTS = [str(req.req) for req in DEV_REQUIREMENTS]


DEPENDENCY_LINKS = []
for file in ["requirements.txt", "requirements.dev.txt"]:
    with open(os.path.join(REQUIREMENTS_DIR, file), "r") as f:
        lines = f.readlines()
        extra_index_urls = filter(lambda line: line.startswith("--extra-index-url"), lines)
        DEPENDENCY_LINKS += extra_index_urls

setup(
    name='commons',
    packages=['commons'],
    install_requires=INSTALL_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS
    },
    dependency_links=DEPENDENCY_LINKS
)

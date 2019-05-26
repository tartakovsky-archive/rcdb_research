import os
from setuptools import setup

REQUIREMENTS_DIR = "requirements"


def clean_requirements(reqs_list):
    return [req.rstrip("\n") for req in reqs_list if req and req != "\n" and not req.startswith("#")]


with open(os.path.join(REQUIREMENTS_DIR, "requirements.txt")) as file:
    INSTALL_REQUIREMENTS = clean_requirements(file)


with open(os.path.join(REQUIREMENTS_DIR, "requirements.dev.txt")) as file:
    DEV_REQUIREMENTS = clean_requirements(file)


setup(
    name='commons',
    packages=['commons'],
    install_requires=INSTALL_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS
    }
)

import os
import sys
import subprocess
from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

REQUIREMENTS_DIR = "requirements"

pip_session = PipSession()


def install(package):
    cmd_args = [sys.executable, '-m', 'pip', 'install', package]
    with subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        print(process.stdout.read())
        print(process.stderr.read())


def parse_reqs(path):
    return [r.requirement for r in parse_requirements(path, session=pip_session)]


INSTALL_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.txt"))
DEV_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.dev.txt"))
SETUP_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.pre.txt"))

for build_req in SETUP_REQUIREMENTS:
    print(f'{build_req} installation...')
    install(build_req)

module_name = 'rcdb_research'
prefix_dev = os.environ.get('DEV_PREFIX')

if prefix_dev:
    new_module_name = f'{prefix_dev}_{module_name}'
    if 'egg_info' in sys.argv:
        os.rename(module_name, new_module_name)

    module_name = new_module_name


setup(
    name=module_name,
    packages=find_packages(include=[f"{module_name}*"]),
    install_requires=INSTALL_REQUIREMENTS + SETUP_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS
    },
)

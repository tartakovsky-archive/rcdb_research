import os
import sys
import subprocess
from shutil import copy2
from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

REQUIREMENTS_DIR = "requirements"

pip_session = PipSession()


def cmd(cmd_args, cwd=None):
    with subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd) as process:
        print(process.stdout.read())
        print(process.stderr.read())


def parse_reqs(path):
    return [r.requirement for r in parse_requirements(path, session=pip_session)]


INSTALL_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.txt"))
DEV_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.dev.txt"))
SETUP_REQUIREMENTS = parse_reqs(os.path.join(REQUIREMENTS_DIR, "requirements.pre.txt"))

for build_req in SETUP_REQUIREMENTS:
    print(f'{build_req} installation...')
    cmd([sys.executable, '-m', 'pip', 'install', build_req])

module_name = 'rcdb_research'
prefix_dev = os.environ.get('DEV_PREFIX')


if prefix_dev:
    new_module_name = f'{prefix_dev}_{module_name}'
    if 'egg_info' in sys.argv:
        os.rename(module_name, new_module_name)

    module_name = new_module_name

# install mc_sizing
if 'egg_info' in sys.argv:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, module_name, 'sizing', 'mc_sizing_cpp')
    cmd(['make'], cwd=source_dir)
    dist_path = os.path.join(base_dir, module_name, 'sizing', 'mc_sizing')
    copy2(os.path.join(source_dir, 'lib.so'), dist_path)


setup(
    name=module_name,
    packages=find_packages(include=[f"{module_name}*"]),
    package_data={f'{module_name}.sizing.mc_sizing': ['*.so']},
    include_package_data=True,
    install_requires=INSTALL_REQUIREMENTS + SETUP_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS
    },
)

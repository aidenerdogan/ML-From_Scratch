from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '1.0.1'

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='ml_fs',
    version=__version__,
    description='Python implementations of some of the fundamental Machine Learning models and algorithms from scratch.',
    url='https://github.com/aidenerdogan/ML-From_Scratch',
    download_url='https://github.com/aidenerdogan/ML-From_Scratch/archive/main.zip',
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    author='Aiden Erdogan',
    install_requires=install_requires,
    setup_requires=['numpy>=1.10', 'scipy>=0.17'],
    dependency_links=dependency_links,
)

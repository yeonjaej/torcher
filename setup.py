from setuptools import setup  # This line replaces 'from setuptools import setup'
import argparse

import io,os,sys
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="torcher",
    version="0.1.0",
    #include_package_data=True,
    author=['Kazuhiro Terao'],
    author_email='kterao@slac.stanford.edu',
    description='Torch utility tools',
    license='MIT',
    keywords='torch',
    project_urls={
        'Source Code': 'https://github.com/drinkingkazu/torcher'
    },
    url='https://github.com/drinkingkazu/torcher',
    #scripts=['bin/run_flow2supera.py'],
    packages=['torcher','torcher/models'],
    package_dir={'': 'src'},
    #package_data={'flow2supera': ['config_data/*.yaml']},
    install_requires=[
        'numpy',
        'torch',
        'h5py',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)

# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
  long_description = fh.read()

packages = [p for p in find_packages() if "third_party" not in p]
print(packages)

setup(
  name='ats-models',
  packages=packages,
  version='0.0.1',
  description='Code to finetune sentence/paragraph based models for ATS',
  author='Annette Rios',
  author_email='rios@cl.uzh.ch',
  url='https://github.com/a-rios/ats-models.git',
  keywords=['ATS'],
  install_requires=['torch', 'transformers', 'numpy'],
  long_description=long_description,
  long_description_content_type='text/markdown',
  classifiers=[
    'Programming Language :: Python :: 3.9',
  ]
)

from setuptools import find_packages,setup
from typing import List


setup(
    name='Fraud_TX',
    version='0.0.1',
    author='Deepraj Arya',
    author_email='mailforarya000@gmail.com',
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)
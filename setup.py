from setuptools import find_packages,setup
from typing import List

"""HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements"""

setup(
    name='Fraud_TX',
    version='0.0.1',
    author='Deepraj Arya',
    author_email='mailforarya000@gmail.com',
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)
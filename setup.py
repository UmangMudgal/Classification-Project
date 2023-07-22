from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> List[str]:
    """
        Descriptionn : This will return List of Requirements
    """

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirments_list = [req.replace('\n','') for req in requirements]

        if HYPHEN_E_DOT in requirments_list:
            requirments_list.remove(HYPHEN_E_DOT)


    return requirments_list


setup(
    name='APS sensor fault Classification',
    version='0.0.1',
    author='Umang Mudgal',
    author_email='mudgal0709@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)


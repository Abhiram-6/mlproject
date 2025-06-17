from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    Returns a list of requirements from the specified file.
    '''
    HYPEN_E_DOT = '-e .'
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

requirements = get_requirements('requirements.txt')

setup(
    name='ml-project',
    version='0.0.1',
    author='abhiram',
    author_email='abhiramgajula9@gmail.com',
    packages=find_packages(),
    install_requires=requirements
)

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT="-e ."

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path, 'r') as libraries:
        for library in libraries:
            library = library.strip()
            if library != HYPHEN_E_DOT:
                requirements.append(library)
    return requirements

setup(
    name='iris_classification_ml_project',
    version='0.0.0',
    author='AnggaDS/Mas brams',
    author_email='anggadwisunarto3@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
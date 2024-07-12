from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT ='-e .' # This is the last line in requirements.txt file which triggers setup.py; we mush remove this from the list
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements needed for this project    
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements
## Setup and User Info:

setup(
name='ShopChat',
description='A RAG based shopping assistance to help buyer on buying product from Amazon [Using ABO dataset]',
version='1.0',
author='Arindam Choudhury',
author_email='arindam.choudhury.email@gmail.com',
url='',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'))
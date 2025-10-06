from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT='-e .'
def get_requirment(file_path:str)->List[str]:
    '''
    this funtion will return the list of requirments
    '''
    
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        [req.replace("\n","") for req in requirments]
        
        if(HYPEN_E_DOT in requirments):
            requirments.remove(HYPEN_E_DOT)
    return requirments
    
    
setup(
    name='mlproject',
    version='0.0.1',
    author='Jatin',
    author_email='jatinprakashrathore@gmail.com',
    packages=find_packages(),
    install_requires=get_requirment('requirments.txt'),
    
)
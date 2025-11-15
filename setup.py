from setuptools import find_packages, setup
from typing import List  # Correct capitalization

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements.txt file and returns a list of packages.
    It ignores lines like '-e .' which are editable installs.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newlines and extra spaces
        requirements = [req.strip() for req in requirements]

        # Remove editable installs
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements 

# Setup function
setup(
    name='CSE',
    version='0.0.1',
    author='HARSH',
    author_email='your_email@example.com',  # Optional
    packages=find_packages(),               # Automatically find packages
    install_requires=get_requirements("requirements.txt"),  # Load dependencies
    python_requires=">=3.8",                # Minimum Python version
)
# setup.py
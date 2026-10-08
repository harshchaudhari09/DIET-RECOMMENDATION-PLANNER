from setuptools import find_packages, setup

setup(
    name="curadiet",
    version="2.0.0",
    description="Chronic disease (diabetes) risk prediction + diet recommendation planner",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.4",
        "flask>=2.3",
        "joblib>=1.3",
    ],
    python_requires=">=3.9",
)

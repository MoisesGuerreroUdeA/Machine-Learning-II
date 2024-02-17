from setuptools import setup, find_packages

setup(
    name="unsupervised",
    version="1.0",
    packages=find_packages(),
    package_dir={'': 'unsupervised'},
    author="Moisés Guerrero",
    author_email="moises.guerrero@udea.edu.co",
    description="Unsupervised package for ML2 implementations"
)
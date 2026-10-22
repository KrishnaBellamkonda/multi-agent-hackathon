from distutils.core import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="activehit",
    version="0.1.0",
    packages=find_packages(),
    author="Andrea Rubbi",
    author_email="andrea.rubbi.98@gmail.com",
    license="GPL-3.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    classifiers=[
        "Operating System :: OS Independent",
    ],
)

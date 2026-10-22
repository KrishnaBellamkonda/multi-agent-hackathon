from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="slingpy",
    version="0.2.11.post1",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
)

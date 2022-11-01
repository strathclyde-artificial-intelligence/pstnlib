from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pstnlib',
    version='0.0.1',
    author='Andrew Murray',
    author_email='a.murray@strath.ac.uk',
    description='Probabilistic Temporal Network Library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/strathclyde-artificial-intelligence/pstnlib',
    license='LICENSE.txt',
    packages=find_packages(),
    install_requires=['antlr4-python3-runtime==4.10', 'scipy', 'numpy', 'gurobipy', 'graphviz', 'otpl'],
    dependency_links=['otpl @ git+ssh://git@github.com/strathclyde-artificial-intelligence/otpl@v1.1#egg=otpl']
)
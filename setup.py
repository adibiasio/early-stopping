from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='early_stopping',
    version='0.0.1',
    packages=find_packages(exclude=('tests')),
    url='https://github.com/adibiasio/early-stopping',
    license='Apache-2.0',
    install_requires=requirements,
    description='Identifying efficient early stopping strategies in iterative learners',
)

from setuptools import setup, find_packages

setup(
    name='gym_cloudsimplus',
    version='0.5.0',
    install_requires=['gym', 'py4j', 'numpy'],
    packages=find_packages()
)

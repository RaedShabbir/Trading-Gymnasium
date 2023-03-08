from setuptools import setup, find_packages


setup(
    name="trading_gym",
    version="0.1",
    packages=find_packages(),
    author="Raed Shabbir",
    install_requires=[
        gymnasium==0.28.0,
        matplotlib==3.7.0,
        numpy==1.23.4,
        pandas==1.5.3,
        setuptools==64.0.2,
    ],
    package_data={"trading_gym": ["datasets/data/*"]},
)

#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Daniel Bedolla Cornejo",
    author_email='daniel.bedollac@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="box_jenkins contains all the tools you need to define an ARIMA model for a time series based on the methodology of Box and Jenkins, moreover it includes ato find an ARIMA m following this methodology automatically.",
    entry_points={
        'console_scripts': [
            'boxjenkins=boxjenkins.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='boxjenkins',
    name='boxjenkins',
    packages=find_packages(include=['boxjenkins', 'boxjenkins.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/dbedollac/boxjenkins',
    version='0.0.0',
    zip_safe=False,
)

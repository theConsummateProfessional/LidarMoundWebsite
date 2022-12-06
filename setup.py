from setuptools import setup, find_packages

setup(
    name='Lidar-Mound-Detector',
    version='0.1.0',
    description='Lidar Mound Recognition Tool',
    author='Ethan Hood',
    author_email='ethanphood@gmail.com',
    packages=find_packages(
        include=[
            'training-utils', 'training-utils.*'
        ]
    ),
    install_requires=[
        'pylas',
        'numpy',
        'pandas',
        'matplotlib',
        'sklearn'
    ]
)
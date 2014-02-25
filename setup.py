import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pywarm',
    version="0.5",
    author='Abraham Lee',
    author_email='tisimst@gmail.com',
    description='Particle swarm optimization (PSO) with constraint support',
    url='https://github.com/tisimst/pyswarm',
    license='BSD License',
    long_description=read('README.rst'),
    packages=['pyswarm'],
    install_requires=['numpy'],
    keywords=[
        'PSO',
        'particle swarm optimization',
        'optimization',
        'python'
        ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
        ]
    )
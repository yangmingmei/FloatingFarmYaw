from setuptools import setup, find_packages

setup(
    name='FloatingFarmYaw',
    version='0.1.0',
    description='A floating offshore wind farm simulation and control framework using FLORIS, MoorPy, and PyTorch',
    author='Mingyang Mei',
    author_email='meimingyang@stu.xjtu.edu.cn',
    url='https://github.com/yangmingmei/FloatingFarmYaw',
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "floris",
        "py_wake",
        "moorpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License Version 2.0",
        "Operating System :: Windows/Linux, Cuda",
    ],
    python_requires='>=3.7',
)

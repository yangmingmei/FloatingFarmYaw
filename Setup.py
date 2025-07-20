from setuptools import setup, find_packages

setup(
    name='your-package-name',  # 🚨 替换为你希望 pip install 使用的名字
    version='0.1.0',
    description='A wind farm control simulation framework using FLORIS, MoorPy, and PyTorch',
    author='Your Name',
    author_email='your@email.com',
    url='https://github.com/yourusername/yourrepo',  # 🚨 替换为你的 GitHub 链接
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
        "License :: OSI Approved :: MIT License",  # 可根据你项目更换开源协议
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

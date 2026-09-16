from setuptools import setup, find_packages

setup(
    name="tensor-lattice-neural-layer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
    author="Aa Baroq",
    description="High-performance deep-tech engine for enterprise tensor/matrix optimization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

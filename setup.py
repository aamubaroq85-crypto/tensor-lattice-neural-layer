import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensor-lattice-neural-layer",
    version="0.1.0",
    author="Baroq",
    author_email="aamubaroq85@gmail.com",
    description="Tensor Lattice Neural Layer for PyTorch memory and latency optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aamubaroq85-crypto/tensor-lattice-neural-layer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
    ],
)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tlnl",
    version="0.1.0",
    author="TLNL Team",
    author_email="author@example.com",
    description="Tensor Lattice Neural Layer for PyTorch memory & latency optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/tlnl",  # Sesuaikan dengan URL repo Anda
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
    ],
)

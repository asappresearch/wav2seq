from io import open
from setuptools import find_packages, setup

setup(
    name="wav2seq",
    version="0.0.1",
    author="Felix Wu",
    author_email="fwu@asapp.com",
    description="Wav2seq",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="speech ssl",
    license="MIT",
    url="",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "torch>=1.0.0",
        "numpy",
        "pandas>=1.0.1",
        "requests",
        "tqdm>=4.31.1",
        "matplotlib",
        "einops",
        "fire",
    ],
    entry_points={},
    include_package_data=True,
    python_requires=">=3.6",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

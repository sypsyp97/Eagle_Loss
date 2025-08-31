from setuptools import setup, find_packages
setup(
    name="eagle-loss",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.7.1-rc1",
        "torchvision",
        "Pillow",
    ],
    python_requires=">=3.11",
    author="Yipeng Sun",
    author_email="yipeng.sun@fau.de",
    description="An Edge-Aware Gradient Localization Enhanced Loss for CT Image Reconstruction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)

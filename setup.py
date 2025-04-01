from setuptools import setup, find_packages

setup(
    name="rearrange",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",  # Optional
        "einops",  # For time comparison
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A flexible array rearrangement library inspired by einops",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vedantasp/sarvam-assignment",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 
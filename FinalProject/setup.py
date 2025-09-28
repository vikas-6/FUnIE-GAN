"""
Setup script for Underwater Image Enhancement for Maritime Security
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="underwater-security",
    version="1.0.0",
    author="Maritime Security Development Team",
    author_email="security@maritime.in",
    description="AI-driven underwater image enhancement for maritime security applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Defense/Security",
        "Programming Language :: Python :: 3",
        "License :: Proprietary",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "PyYAML>=5.3.0",
    ],
)
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="decologr",
    version="0.1.0",
    author="Mark Brooks",
    author_email="",
    description="Decorative Logger - A logging utility with emoji decorations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mxflask/decologr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "numpy": ["numpy>=1.20.0"],
    },
)


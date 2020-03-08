from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="torch_xtend",
    version="0.1",
    author="leengsmile",
    author_email="leengsmile@126.com",
    description="PyTorch Extensions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    find_packages=find_packages(exclude=["tests"]),
    url="https://github.com/leengsmile/pytorch-xtend",
    python_requires=">=3.6",
    license="MIT",
    keywords=["PyTorch", "deep learning"]
)

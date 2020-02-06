import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zeustools-NanoExplorer", # Replace with your own username
    version="0.0.2",
    author="Christopher Rooney",
    author_email="ctr44@cornell.edu",
    description="A compendium of tools useful for processing ZEUS-2 data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NanoExplorer/zeustools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

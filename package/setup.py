from setuptools import setup, find_packages

setup(
    name="ve_sim",
    version="0.1.0",
    packages=find_packages(include=["ve_sim", "ve_sim.*"]),
    install_requires=[
    "traci",
    "simpy",
    "numpy",
    "torch",
    "gymnasium",
    "pandas"
    ],
    entry_points={
        "console_scripts": [
            "run=ve_sim.main:run",
        ],
    },
    author=" ",
    description="package!", # add description
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/...", # add git 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.7',
    )

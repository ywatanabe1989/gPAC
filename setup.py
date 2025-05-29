from setuptools import setup, find_packages

setup(
    name="gpac",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "torchaudio>=0.9.0",
    ],
)
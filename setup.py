from setuptools import setup, find_packages

setup(
    name="cognimem",
    version="0.1.0",
    description="CogniMem: A cognitive memory architecture for AI Agents",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "sentence-transformers>=2.2.0",
        "spacy>=3.7.0",
    ],
    python_requires=">=3.8",
)
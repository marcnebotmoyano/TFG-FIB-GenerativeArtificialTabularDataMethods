from setuptools import setup, find_packages

setup(
    name="gatdm",
    version="0.1.0",
    author="Marc Nebot i Moyano",
    author_email="marcnebotmoyano@gmail.com",
    description="Una librería para la generación de datos artificiales tabulares",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/marcnebotmoyano/TFG-FIB-GenerativeArtificialTabularDataMethods",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "torch",
        "seaborn",
    ],
)

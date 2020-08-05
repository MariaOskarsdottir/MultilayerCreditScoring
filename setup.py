import setuptools
with open("README.md", "r") as fh:
    long_descripiton = fh.read()

setuptools.setup(
    name="multilayer-credit-scoring",
    version="0.0.8",
    author="Sigurjon Thorsteinsson",
    author_email="grjoni80@gmail.com",
    description="Python implementation of the MultilayerCreditScoring algorithm",
    long_description=long_descripiton,
    long_description_content_type="text/markdown",
    url="https://github.com/MariaOskarsdottir/MultilayerCreditScoring",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.19",
        "networkx==2.3",
        "scipy==1.5",
        "scikit-learn==0.23" ,
        # "multinetx @ git+ssh://git@github.com/nkoub/multinetx.git#egg=multinetx-2.3" # looks like dependancy like this is not allowed to be uploaded on PyPI
    ]
)
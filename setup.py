import setuptools
with open("README.md", "r") as fh:
    long_descripiton = fh.read()

setuptools.setup(
    name="MuLP",
    version="0.0.5",
    author="Sigurjon Thorsteinsson, Emiliano Penaloza",
    author_email="grjoni80@gmail.com, emilianopp550@gmail.com",
    description="Python implementation of the Multilayer Personalized Page Rank algorithm",
    long_description=long_descripiton,
    long_description_content_type="text/markdown",
    url="https://github.com/Banking-Analytics-Lab/MuLP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    setup_requires=["numpy","scikit-learn","igraph"], 
    install_requires=[
        "scikit-learn",
        "numpy",
        "scipy",
        "igraph",
        "pandas"
        ]
)


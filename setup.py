from setuptools import setup, find_packages


authors = ""
correspond_author_email = ""
description = ""

with open("README.md", "r") as fh:
     long_description = fh.read()

setup(
    name="cfos_app",
    author=authors,
    author_email=correspond_author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.5",
        "pandas==1.4.3",
        "opencv-python==4.6.0.66",
        "path==16.2.0",
        "tqdm==4.62.1",
        "nibabel==3.2.1",
        "tifffile==2022.4.8",
        "connected-components-3d==3.10.0",
        "monai==0.7.0",
        "scipy==1.8.1",
        "torch==1.11.0",
        "torchaudio==0.11.0",
        "torchvision==0.12.0",
        "scikit-image==0.18.1",

    ],
    extras_require={},
    version="1.0",
    cmdclass={},
)
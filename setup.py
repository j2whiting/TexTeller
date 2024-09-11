from setuptools import setup, find_packages


# Define the base dependencies
install_requires = [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "nltk",
    "python-multipart",
    "shapely",
    "pyclipper",
    "optimum[exporters]",
    "opencv-python-headless==4.9.0.80",
]

setup(
    name="texteller",
    version="0.1.2",
    author="OleehyO",
    author_email="1258009915@qq.com",
    description="A meta-package for installing dependencies",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/j2whiting/TexTeller",
    packages=find_packages(include=["texteller", "texteller.*"]),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

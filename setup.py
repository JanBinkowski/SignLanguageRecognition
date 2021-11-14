from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.18'
DESCRIPTION = 'Sign Language Recognition tool.'
LONG_DESCRIPTION = 'Sign Language Recognition tool. It works in real time using additionally OpenCV and Mediapipe libraries.'

# Setting up
setup(
    name="SignLanguageRecognition",
    version=VERSION,
    author="Jan Binkowski",
    author_email="<jan.binkowski@wp.pl>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/JanBinkowski/SignLanguageRecognition",
    project_urls={
        "Bug Tracker": "https://github.com/JanBinkowski/SignLanguageRecognition",
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['opencv-python', 'mediapipe', 'numpy', 'tensorflow'],
    keywords=['python', 'sign language', 'sign language recognition', 'recognition in real time', 'action recognition'],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    include_package_data=True,
)
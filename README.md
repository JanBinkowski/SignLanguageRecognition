# Sign Language Recognition tool
[![GitHub License](https://img.shields.io/github/license/JanBinkowski/SignLanguageRecognition?style=plastic)](https://github.com/JanBinkowski/SignLanguageRecognition/blob/master/LICENSE)  [![GitHub Stars](https://img.shields.io/github/stars/JanBinkowski/SignLanguageRecognition?style=plastic)](https://github.com/JanBinkowski/SignLanguageRecognition/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/JanBinkowski/SignLanguageRecognition?style=plastic)](https://github.com/JanBinkowski/SignLanguageRecognition/issues)
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [How it was made](#how-it-was-made) 
* [Setup](#setup)
* [Getting started](#getting-started)
* [License](#license)

---

## General info
SignLanguageRecognition package is a opensource tool to estimate sign language from camera vision. This project is a part of   my Bachelor Thesis and contains the implementation of sign language recognition tool using a LSTM Neural Network, TensorFlow Keras and other opensorce libraries like: OpenCV or MediaPipe.
Here is a link leading to PyPi package repository: [pypi.org/SignLanguageRecognition](https://pypi.org/project/SignLanguageRecognition/).
	
---	
	
## Technologies
Project is created with:
* **Python**: 3.8.8
* **OpenCV-Python**: 4.5.3.56
* **TensorFlow**: 2.4.1
* **MediaPipe**: 0.8.7.3
* **NumPy**: 1.19.5

---	

## How it woas made
### Creating a dataset and model training
The entire dataset used to model training was created by me from scratch. Gathering the data 
was made by using opensource libraries: OpenCV and Mediapipe. Dataset is collection of 100 thirty-frame sequences for each class
and every frame is saved as Numpy array which came as the output of the mediapipe library.

Proccess of collecting data is demonstrated right below:

| class: a            |  class: b |  class: c |
:-------------------------:|:-------------------------:|:-------------------------:
![a_gif.gif](FILES_FOR_README/a_gif.gif) |  ![b_gif.gif](FILES_FOR_README/b_gif.gif) | ![c_gif.gif](FILES_FOR_README/c_gif.gif)

To train the model, the TensorFlow library was used with LSTM layers. The whole training
process is available [here](https://github.com/JanBinkowski/SignLanguageRecognition/blob/master/SignLanguageRecognitionLearning/SignLanguageRecognition.ipynb).


---

## Setup 
All package details you can find here: [pypi.org/SignLanguageRecognition](https://pypi.org/project/SignLanguageRecognition/).
The Python Package Index (PyPI) is a repository of software for the Python programming language.
Make sure that you have installed ```python``` (>=3.6) and you can run ```python``` from the command line. Check it by running:
```
python --version
```
Now you need to use a ```pip``` to conduct installation process. ```pip``` is a package management system used to install and manage software packages/libraries written in Python. These files are stored in a large “on-line repository” termed as Python Package Index (PyPI).  
To check if ```pip``` is already installed on your system, just go to the command line and execute the following command:
```
pip -V
```
After you make sure you can run ```pip``` from the command line you should ensure that ```pip``` version is up-to-date. To check this use command below:
```
pip install --upgrade pip
```
To run SignLanguageRecognition package, install you have to install it locally using ```pip```:
```
pip install SignLanguageRecognition
```
 or to install specific version, for example 0.0.17:
 ```
 pip install SignLanguageRecognition==0.0.17
 ```

**Note.** When you are installing this library, the following are also installed as required: ```opencv-python```,```mediapipe```,```numpy``` and ```tensorflow```.
 
More details about installing and running python packages here: [Installing Packages (python.org)](https://packaging.python.org/tutorials/installing-packages/).

---

## Getting started
This section shows example how to use SignLanguageRecognition library.
First of all you have to open Python interpreter on your device. An example way to do this is by typing a command:
```
python
```
After that, the Python intrpreter will be opened. Then import a method from previously installed package:
```python
>> from SignLanguageRecognition import signLanguageRecognizer
>> signLanguageRecognizer.signLanguageRecognizerMethod()
```
This code will cause opening a new OpenCV window with ready-to-work tool.
Enjoy testing!

---

## License
>You can check out the full license [here](https://github.com/JanBinkowski/SignLanguageRecognition/blob/master/LICENSE)

This project is licensed under the terms of the **MIT** license.
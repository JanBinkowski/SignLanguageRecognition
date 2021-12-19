from SignLanguageRecognitionLearning.createDataset import *
from SignLanguageRecognitionLearning.signLanguageRecognizer import *

def handleActionInMain():
    while(True):
        _input = input('1 - Create new class \n2 - Sign Language Recognizer\n4 - Make testing\nq - Quit\n')
        if(_input == '1'):
            createDataset()
        if (_input == '2'):
            signLanguageRecognizer()
        if (_input == '3'):
            signLanguageRecognizer_2()
        elif(_input == 'q'):
            print('\nLeaving a program...')
            break
        else:
            pass

def main():
    handleActionInMain()

if __name__ == "__main__":
    main()

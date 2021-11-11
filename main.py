from createDataset import *
from signLanguageRecgnizer import *

def handleActionInMain():
    while(True):
        _input = input('1 - Create new class \n2 - Sign Language Recognizer\nq - Quit\n')
        if(_input == '1'):
            createDataset()
        if (_input == '2'):
            signLanguageRecognizer()
        elif(_input == 'q'):
            print('\nLeaving a program...')
            break
        else:
            pass

def main():
    handleActionInMain()

if __name__ == "__main__":
    main()

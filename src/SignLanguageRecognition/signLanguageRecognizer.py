import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pkg_resources

DATA_PATH = os.path.join('../MP_DATA')
DATA_PATH_VIDEO = os.path.join('../MP_VIDEOS')
actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z'])
no_sequences = 100
sequence_length = 30

mpHolistic = mp.solutions.holistic
mpDrawing = mp.solutions.drawing_utils

def mediapipeDetection(image_param, model):
    image = cv2.cvtColor(image_param, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def drawLandmarks(image, results):
    # mpDrawing.draw_landmarks(image, results.face_landmarks, mpHolistic.FACEMESH_CONTOURS,
    #                          mpDrawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    #                          mpDrawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mpDrawing.draw_landmarks(image, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
                             mpDrawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                             mpDrawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mpDrawing.draw_landmarks(image, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
                             mpDrawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                             mpDrawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # mpDrawing.draw_landmarks(image, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS,
    #                          mpDrawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
    #                          mpDrawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extractKeypoints(results):
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # print(np.concatenate([lh, rh]))
    # return np.concatenate([pose, face, lh, rh])
    return np.concatenate([lh, rh])

def signLanguageRecognizerMethod():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    # model.load_weights('action.h5')
    model.load_weights(pkg_resources.resource_filename('SignLanguageRecognition', 'action.h5'))
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.2
    cap = cv2.VideoCapture(0)
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipeDetection(frame, holistic)

            # if(results.left_hand_landmarks):
            #     print(results.left_hand_landmarks.landmark[1])
            # print(results.left_hand_landmarks)

            drawLandmarks(image, results)
            keypoints = extractKeypoints(results)

            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = model.predict(tf.expand_dims(sequence, axis=0))[0]
                # print(np.expand_dims(sequence, axis=0))
                predictions.append(np.argmax(res))
                # clearConsole()
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if (max(res) >= threshold):
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                        cv2.putText(image, actions[np.argmax(res)]+' : '+str((max(res) * 100).astype(float))+' %',
                                    (15, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("window", image)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("\nEscape hit, closing...")
                break

        cap.release()
        cv2.destroyAllWindows()



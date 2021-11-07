import cv2
import numpy as np
import mediapipe as mp


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


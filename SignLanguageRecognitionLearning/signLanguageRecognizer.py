from SignLanguageRecognitionLearning.landmarkDetection import *

def signLanguageRecognizer():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.load_weights('C://Users//JanBinkowski//Desktop//SignLanguageRecognition//Weights//action.h5')
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.2
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipeDetection(frame, holistic)

            # if(results.left_hand_landmarks):
            #     print(results.left_hand_landmarks.landmark[1])
            # print(results.left_hand_landmarks)

            # drawLandmarks(image, results)
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
                            # print(sentence)
                        # print(actions[np.argmax(res)], ' <===> Probability: {}%'.format((max(res) * 100).astype(float)))
                        cv2.putText(image, actions[np.argmax(res)]+' : '+str((max(res) * 100).astype(float))+' %',
                                    (15, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    # clearConsole()

            cv2.imshow("window", image)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("\nEscape hit, closing...")
                break

        cap.release()
        cv2.destroyAllWindows()


from SignLanguageRecognitionLearning.landmarkDetection import *
from decimal import Decimal

def signLanguageRecognizer():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    # model.load_weights('C://Users//JanBinkowski//Desktop//SignLanguageRecognition//Weights//action.h5')
    # model.load_weights('C://Users//JanBinkowski//Downloads//2021.12.06-18.07-batch_128//Weights//2021.12.06-18.32//action.h5')
    model.load_weights('C://Users//JanBinkowski//Desktop//najlepsze_treningi_sieci//2021.12.07-20.12-batch_128_ale_1000epok_inna_proporcja_danych//Weights//2021.12.07-21.16//action.h5')
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
                ind = np.argpartition(res, -3)[-3:]
                predictions.append(np.argmax(res))

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
                        print(actions[ind[0]]+':'+str((res[ind[0]] * 100).astype(float))+' %'+'      '
                              +actions[ind[1]]+':'+str((res[ind[1]] * 100).astype(float))+' %'+'      '
                              +actions[ind[2]]+':'+str((res[ind[2]] * 100).astype(float))+' %')

            cv2.imshow("window", image)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

        cap.release()
        cv2.destroyAllWindows()


def signLanguageRecognizer_2():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    # model.load_weights('action.h5')
    model.load_weights('C://Users//JanBinkowski//Desktop//najlepsze_treningi_sieci//2021.12.07-20.12-batch_128_ale_1000epok_inna_proporcja_danych//Weights//2021.12.07-21.16//action.h5')
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.2
    string = ''
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipeDetection(frame, holistic)
            drawLandmarks(image, results)
            cv2.putText(image, 'Tap a spacebar to start recognizing.',
                        (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow("window", image)

            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("\nEscape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                start = time.time()
                for frame_num in range(every_sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipeDetection(frame, holistic)
                    drawLandmarks(image, results)

                    cv2.imshow("window", image)
                    keypoints = extractKeypoints(results)

                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                    if len(sequence) == 30:
                        res = model.predict(tf.expand_dims(sequence, axis=0))[0]
                        ind = np.argpartition(res, -3)[-3:]
                        predictions.append(np.argmax(res))
                        if (max(res) >= threshold):
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                            cv2.putText(image, actions[np.argmax(res)] + ' : ' + str(
                                (max(res) * 100).astype(float)) + ' %',
                                        (15, 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                            if len(sequence) == 30:
                                output = actions[ind[0]] + ':' + str((res[ind[0]] * 100).astype(float)) + ' %' + '      '+ actions[ind[1]] + ':' + str((res[ind[1]] * 100).astype(float)) + ' %' + '      '+ actions[ind[2]] + ':' + str((res[ind[2]] * 100).astype(float)) + ' %' +'\n'
                                for i in range(0,23):
                                    if i < 22:
                                        string += str('{0:.5f}'.format((Decimal(res[i] * 100))) + ',')
                                    elif i == 22:
                                        string += str('{0:.5f}'.format((Decimal(res[i] * 100))) + '\n')


                                with open('test_output.txt', 'a') as file:
                                    file.write(output)
                                    print(output)
                                with open('test_output_CSV.txt', 'a') as file:
                                    file.write(string)
                                    print(string)
                                    string = ''
                        sequence=[]
                    cv2.waitKey(1)

                stop = time.time()
                print("\nRecognizing ended. Time: {}\n".format(stop - start))

            # if (cv2.getWindowProperty("window", cv2.WND_PROP_VISIBLE) < 1) or (cv2.waitKey(1) & 0xFF == 27):
            #     break

        cap.release()
        cv2.destroyAllWindows()

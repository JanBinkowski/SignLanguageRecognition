from SignLanguageRecognitionLearning.landmarkDetection import *

def createNewClassDir(className):
    for sequence in range(number_of_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, className, str(sequence)))
            os.makedirs(os.path.join(DATA_PATH_VIDEO, className, str(sequence)))
        except:
            pass

def createDataset():
    classDirName = input('\nClass name: ')
    print('\nOpenCV is starting...')
    createNewClassDir(classDirName)
    cap = cv2.VideoCapture(0)
    seq_counter = 0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipeDetection(frame, holistic)
            drawLandmarks(image, results)
            cv2.putText(image, 'Tap a spacebar to start recording.',
                        (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow("window", image)
            out = cv2.VideoWriter(
                os.path.join(DATA_PATH_VIDEO, classDirName, str(seq_counter), "{}.mp4".format(seq_counter)),
                cv2.VideoWriter_fourcc(*'mp4v'), 15.0,
                (frame_width, frame_height))
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("\nEscape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                start=time.time()
                for frame_num in range(every_sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipeDetection(frame, holistic)
                    drawLandmarks(image, results)
                    cv2.putText(image, 'Class Name: {}. Video Number: {}'.format(classDirName, seq_counter+1),
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    out.write(image)

                    cv2.imshow("window", image)
                    keypoints = extractKeypoints(results)


                    npy_path = os.path.join(DATA_PATH, classDirName, str(seq_counter), str(frame_num))
                    np.save(npy_path, keypoints)

                    print("{}".format(npy_path))
                    # print(keypoints)
                    if(frame_num == every_sequence_length-1):
                        seq_counter += 1
                    if (seq_counter == number_of_sequences):
                        break
                    cv2.waitKey(1)

                stop = time.time()
                print("\n\n\nCollecting frames ended. Time: {}\n\n\n".format(stop-start))


            # if (cv2.getWindowProperty("window", cv2.WND_PROP_VISIBLE) < 1) or (cv2.waitKey(1) & 0xFF == 27):
            #     break

        cap.release()
        cv2.destroyAllWindows()


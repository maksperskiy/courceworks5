# pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib

import cv2
import numpy as np
import os
import time
import mediapipe as mp


def make_dirs(data_path, actions, no_sequences):
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(data_path, action, str(sequence)))
            except:
                pass


def only_hands(mp_hands, mp_drawing, mp_drawing_styles,
               flip=False, threshold=0.5, static_mode=False, camera=0):
    cap = cv2.VideoCapture(camera)
    p_time = 0
    with mp_hands.Hands(
            min_detection_confidence=threshold,
            min_tracking_confidence=threshold, static_image_mode=static_mode) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            if flip:
                image = cv2.flip(image, 1)

            # FPS counter
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            cv2.putText(image, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 245, 215), 3)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.namedWindow('Hands', cv2.WINDOW_NORMAL)
            cv2.imshow('Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


def extract_keypoints(results):
    try:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            lh = np.array([[res.x, res.y, res.z] for res in
                           results.multi_hand_landmarks[idx].landmark]).flatten() \
                if results.multi_handedness[idx].classification[0].label == 'Left' else np.zeros(21 * 3)
            rh = np.array([[res.x, res.y, res.z] for res in
                           results.multi_hand_landmarks[idx].landmark]).flatten() \
                if results.multi_handedness[idx].classification[0].label == 'Right' else np.zeros(21 * 3)
        return np.concatenate([lh, rh])
    except:
        return np.concatenate([np.zeros(21 * 3), np.zeros(21 * 3)])


def get_keypoints(mp_hands, mp_drawing, mp_drawing_styles,
                  actions, no_sequences, sequence_length, data_path,
                  threshold=0.5, static_mode=False, camera=0):
    cap = cv2.VideoCapture(camera)
    p_time = 0
    with mp_hands.Hands(
            min_detection_confidence=threshold,
            min_tracking_confidence=threshold,
            static_image_mode=static_mode) as hands:
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue

                    # FPS counter
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                    cv2.putText(image, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 245, 215), 3)

                    # Make detections
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)

                    # Draw landmarks
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                    # NEW Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 245, 215), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 245, 215), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(1500)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 245, 215), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(data_path, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(5) & 0xFF == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        return



if __name__ == "__main__":
    mp_h = mp.solutions.hands  # Holistic model
    mp_d = mp.solutions.drawing_utils  # Drawing utilities
    mp_ds = mp.solutions.drawing_styles

    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('OGO_Data')

    # Actions that we try to detect
    ACTIONS = np.array([
                            'up'
                          , 'down'
                          , 'left'
                          , 'right'
                          , 'ok'
                          , 'back'
                        ])

    # Thirty videos worth of data
    NO_SEQUENCES = 30

    # Videos are going to be 30 frames in length
    SEQUENCE_LENGTH = 40


    # only_hands(mp_h, mp_d, mp_ds)
    # make_dirs(os.path.join('DATA_PATH'), actions=ACTIONS, no_sequences=5)
    get_keypoints(mp_h, mp_d, mp_ds, ACTIONS, no_sequences=5, sequence_length=5, data_path=os.path.join('DATA_PATH'))

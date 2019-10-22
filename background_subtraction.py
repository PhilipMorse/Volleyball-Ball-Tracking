import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import time
import csv

MODEL = "0.9818182-1571182058"  # Model name to be used (.h5 format)
VID_NAME = "vid1"  # Video file found in vids/
VID_FORMAT = ".mp4"  # Video format

SAVE_OUTPUT = False  # Toggle whether to save the locations of the ball or not

BALL_TRAIL_FRAMES = 20  # Number of frames that the trail stays behind the ball
SHOW_TRAIL = True  # Whether to show a trail or not

SHOW_VIDEO = True  # Whether video is displayed or not
WAIT_KEY = 1  # Speed of video playback, 0 to manually advance

CREATE_TRAINING_DATA = False  # Used to create training set in data/
TRAINING_DATA_FOLDER = "mikasa"  # Folder for training data

# Tentative decent results with the default settings
KERNEL_RESOLUTION = 200  # Higher = More objects detected in frame (Default = 200)
MAX_CONTOURS = 100  # Limit the number of contours for performance (Default = 100)
CLASSIFICATION_THRESHOLD = 0.2  # 0-1 Which contours to evaluate for predictions (Default = 0.2)

# Leave the following on True for proper usage of script
RUN_VIDEO = True  # Run Main Loop
PREDICT_BALL_LOCATION = True  # Turn on to keep ball location predictions history
USE_IMAGE_CLASSIFICATION_MODEL = True  # Determine whether to used image classification model

cap = cv2.VideoCapture("vids/" + VID_NAME + VID_FORMAT)
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
num_frames_max = cap.get(cv2.CAP_PROP_FRAME_COUNT)
kernel_size = int(cap_width / KERNEL_RESOLUTION)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

if USE_IMAGE_CLASSIFICATION_MODEL:
    model = load_model('models/' + MODEL + '.h5')
else:
    model = None


def clean_trajectories(traj):
    output = [[traj[0][0], traj[0][1]]]

    for i, t in enumerate(traj[1:]):
        diff = int(t[2] - traj[i][2])
        if diff == 1:
            output.append([t[0], t[1]])
        else:
            dx = (t[0] - traj[i][0]) / diff
            dy = (t[1] - traj[i][1]) / diff
            for j in range(1, diff + 1):
                output.append([int(traj[i][0] + (dx * j)), int(traj[i][1] + (dy * j))])
    output = np.array(output, dtype=np.int32)
    return output


# TODO: Improve prediction code
class Prediction:
    def __init__(self, frame_n, x, y):
        self.hist = [[frame_n, x, y, 10000, 1000]]  # [frame_n, x, y, dx, dy]

    def new_point(self, frame_n, x, y):
        self.hist.append([
            frame_n,
            x,
            y,
            (x - self.hist[-1][1]) / (frame_n - self.hist[-1][0]),
            (y - self.hist[-1][2]) / (frame_n - self.hist[-1][0])
        ])

    def predict(self, frame_n):
        return [
            self.hist[-1][1] + (self.hist[-1][3] * (frame_n - self.hist[-1][0])),
            self.hist[-1][2] + (self.hist[-1][4] * (frame_n - self.hist[-1][0]))
        ]

    def confidence(self, frame_n):
        return (((2 * ((self.hist[-1][3] ** 2) + (self.hist[-1][4] ** 2) ** 0.5)) ** (
                frame_n - self.hist[-1][0])) + 100)


def classify_contours(contour_list, target_frame):
    probs = []
    for c in contour_list:
        x, y, w, h = cv2.boundingRect(c)
        roi = target_frame[y: y + h, x: x + w]
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (28, 28))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        probability = model.predict_proba(img)
        probs.append(probability[0][1])
        # cv2.putText(frame, str(probability[0][1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return probs


def distance(arr_a, arr_b):
    return ((arr_a[0] - arr_b[0]) ** 2 + (arr_a[1] - arr_b[1]) ** 2) ** 0.5


output = []
if SHOW_TRAIL:
    output_realtime = []
output_img = None
prediction = Prediction(0, cap_width / 2, cap_height / 2)
if RUN_VIDEO:
    while True:
        ret, frame = cap.read()
        frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        fgmask = fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        opening = cv2.dilate(opening, kernel, iterations=1)
        _, thresh = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY)

        if frame_n > 1:
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                if len(contours) < MAX_CONTOURS:
                    if CREATE_TRAINING_DATA:
                        for i, c in enumerate(contours):
                            x, y, w, h = cv2.boundingRect(c)
                            if w > 10:
                                if h > 10:
                                    roi = frame[y: y + h, x: x + w]
                                    filename = 'data/' + TRAINING_DATA_FOLDER + '/' + VID_NAME + '-' + \
                                               str(frame_n) + '-' + str(i) + '.png'
                                    cv2.imwrite(filename, roi)
                    if USE_IMAGE_CLASSIFICATION_MODEL:
                        probabilities = classify_contours(contours, frame)

                        # TODO: Change this naive ball finding
                        if PREDICT_BALL_LOCATION:
                            predicted_location = prediction.predict(frame_n)
                            predicted_confidence = prediction.confidence(frame_n)
                            potentials = []
                            for i, p in enumerate(probabilities):
                                x, y, w, h = cv2.boundingRect(contours[i])
                                dist = distance(predicted_location, [x + w / 2, y + h / 2])
                                if dist < 1:
                                    dist = 1
                                if dist < predicted_confidence:
                                    if p > CLASSIFICATION_THRESHOLD:
                                        cv2.putText(frame, str(dist), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (0, 255, 255))
                                        potentials.append(p / dist)

                                    else:
                                        potentials.append(0)
                                else:
                                    potentials.append(0)
                            max_value = max(potentials)
                            if max_value > 0:
                                max_potential = potentials.index(max_value)
                                x, y, w, h = cv2.boundingRect(contours[max_potential])
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                prediction.new_point(frame_n, x + w / 2, y + h / 2)
                                print(x, y)
                                output.append([x + w / 2, y + h / 2, frame_n])
                                if SHOW_TRAIL:
                                    output_realtime = clean_trajectories(output[-BALL_TRAIL_FRAMES:])
                                # TODO: Implement Machine Learning Training
                            else:
                                x = int(predicted_location[0])
                                y = int(predicted_location[1])
                                cv2.rectangle(frame, (x, y), (x + 10, y + 10), (0, 0, 255), 2)
                                prediction.new_point(frame_n, x, y)
                            if SHOW_TRAIL:
                                cv2.polylines(frame, [output_realtime], 0, (0, 0, 255), 3)


                        else:
                            max_index = probabilities.index(max(probabilities))
                            x, y, w, h = cv2.boundingRect(contours[max_index])
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    else:
                        c = max(contours, key=cv2.contourArea)
                        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if SHOW_VIDEO:
            if frame_n == 1:
                output_img = frame
            cv2.imshow('original', frame)
            cv2.imshow('opened', opening)

            k = cv2.waitKey(WAIT_KEY)
            if k == 27:
                break
            if frame_n == num_frames_max:
                break

cap.release()
cv2.destroyAllWindows()

if SAVE_OUTPUT:
    with open("output/" + VID_NAME + "-" + str(int(time.time())) + ".csv", "a", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(output)
    cv2.imwrite("output/" + VID_NAME + "-" + str(int(time.time())) + ".png", output_img)

import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

RUN_VIDEO = True  # Run Main Loop
USE_IMAGE_CLASSIFICATION_MODEL = True  # Determine whether to used image classification model
MODEL = "0.9818182-1571182058"  # Model name to be used (.h5 format)
CREATE_TRAINING_DATA = False  # Used to create training set in data/
VID_NAME = "vid1"  # Video file found in vids/
VID_FORMAT = ".mp4"  # Video format
KERNEL_RESOLUTION = 200  # Higher = More objects detected in frame
MAX_CONTOURS = 100  # Limit the number of contours for performance
PREDICT_BALL_LOCATION = False  # Turn on to keep ball location predictions history
CLASSIFICATION_THRESHOLD = 0  # 0-1 Which contours to evaluate for predictions

cap = cv2.VideoCapture("vids/" + VID_NAME + VID_FORMAT)
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
kernel_size = int(cap_width / KERNEL_RESOLUTION)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

model = None

if USE_IMAGE_CLASSIFICATION_MODEL:
    model = load_model('models/' + MODEL + '.h5')


# TODO: Improve prediction code
class Prediction:
    def __init__(self, frame_n, x, y):
        self.count = 1
        self.hist = [[frame_n, x, y, 0, 0]]  # [frame_n, x, y, dx, dy]

    def new_point(self, frame_n, x, y):
        self.hist.append([
            frame_n,
            x,
            y,
            (x - self.hist[-1][1]) / (frame_n - self.hist[-1][0]),
            (y - self.hist[-1][2]) / (frame_n - self.hist[-1][0])
        ])
        self.count += 1

    def predict(self, frame_n):
        return [
            self.hist[-1][1] + (self.hist[-1][3] * (frame_n - self.hist[-1][0])),
            self.hist[-1][2] + (self.hist[-1][4] * (frame_n - self.hist[-1][0]))
        ]


    def count(self):
        return self.count


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
        cv2.putText(frame, str(probability[0][1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return probs


def distance(arr_a, arr_b):
    return ((arr_a[0] - arr_b[0]) ** 2 + (arr_a[1] - arr_b[1]) ** 2) ** 0.5

active_prediction = 1
predictions = [Prediction(0,cap_width/4,cap_height/2), Prediction(0,cap_width/2,cap_height/2), Prediction(0,cap_width*3/4,cap_height/2)]
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
                                    filename = 'data/' + VID_NAME + '-' + str(frame_n) + '-' + str(i) + '.png'
                                    cv2.imwrite(filename, roi)
                    if USE_IMAGE_CLASSIFICATION_MODEL:
                        probabilities = classify_contours(contours, frame)

                        max_index = probabilities.index(max(probabilities))
                        x, y, w, h = cv2.boundingRect(contours[max_index])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        # TODO: Change this naive ball finding
                        if PREDICT_BALL_LOCATION:
                            predicted_locations = [predictions[0].predict(frame_n),predictions[1].predict(frame_n),predictions[2].predict(frame_n)]
                            potentials = []
                            for i, p in enumerate(probabilities):
                                if p>CLASSIFICATION_THRESHOLD:
                                    x, y, w, h = cv2.boundingRect(contours[i])
                                    potentials.append([
                                        i,
                                        p,
                                        distance(predicted_locations[0],[x+w/2,y+h/2]),
                                        distance(predicted_locations[1],[x+w/2,y+h/2]),
                                        distance(predicted_locations[2],[x+w/2,y+h/2])
                                    ])
                                #TODO: Implement Machine Learning Training
                    else:
                        c = max(contours, key=cv2.contourArea)
                        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('original', frame)
        cv2.imshow('opened', opening)

        k = cv2.waitKey(1)
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()

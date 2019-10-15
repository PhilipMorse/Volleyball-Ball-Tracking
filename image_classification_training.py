from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import time

TEST_MODEL = False

train = pd.read_csv('data/annotated_data/annotations.csv')
train_images = []

for i in tqdm(range(train.shape[0])):
    img = image.load_img('data/annotated_data/' + train['id'][i], target_size=(28, 28, 3), grayscale=False)
    img = image.img_to_array(img)
    img = img / 255
    train_images.append(img)

X = np.array(train_images)

y = train['label'].values
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test,y_test))

try:
    model.save('models/' + str(history.history.get('accuracy')[-1]) + "-" + str(int(time.time())) + '.h5')
    print("Model Saved: " + str(history.history.get('accuracy')[-1]) + "-" + str(int(time.time())) + '.h5')
except:
    print("Something went wrong")

if TEST_MODEL:
    test_image = []
    for file in glob.glob("data/test_data/*.png"):
        img = image.load_img(file, target_size=(28, 28, 3), grayscale=False)
        img = image.img_to_array(img)
        img = img / 255
        test_image.append(img)

    test = np.array(test_image)
    prediction = model.predict_classes(test)

    print(prediction)


# Volleyball Ball Tracking:
This code is used to train and use software to track a beach volleyball using Neural Networks and Open CV

![Imgur](https://i.imgur.com/b7uW12T.png)


# How to:
**background_subtraction.py**

This file is used to run through a video.

**annotate_images.py**

This file is used to manually determine whether each obtained image is a ball or not to create the testing and training sets

**create_refs.py**

This file is used to manually select regions that have a ball in a video

**image_classification_training.py**

This file is used to create a keras model to be used in background_subtraction.py

**ball_trajectory.py**

This file prints an image with the tracked trajectory previously obtained.

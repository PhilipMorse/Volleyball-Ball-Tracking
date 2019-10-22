import cv2
import csv
import numpy as np
import random

FILENAME = "vid3-1571431556"
THRESHOLD_ANGLE = 30


def separate_trajectories(traj):
    changes = [1]
    vectors = []
    for i in range(len(traj)-4):
        vectors.append([traj[i+4][0]-traj[i][0],traj[i+4][1]-traj[i][1]])
    vectors = np.array(vectors)
    angles = np.degrees(np.arctan2(*vectors.T[::-1])) % 360.0
    last = 0
    for i,a in enumerate(angles[:-1]):
        if i>last+4:
            if abs(a-angles[i+1]) > THRESHOLD_ANGLE:
                if abs(a - angles[i + 1]) < 360 - THRESHOLD_ANGLE:
                    changes.append(i+4)
                    last = i
    changes.append(len(traj))
    return changes

def clean_trajectories(traj):
    output = np.array(traj, dtype=np.int32)
    output = np.delete(output, 2, axis=1)

    return output


def load_trajectories():
    traj = []
    with open("output/" + FILENAME + ".csv") as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            traj.append(row)
    return traj


if __name__ == "__main__":
    image = cv2.imread("output/" + FILENAME + ".png")

    trajectories = load_trajectories()
    changes = separate_trajectories(trajectories)
    trajectory = clean_trajectories(trajectories)
    for i,c in enumerate(changes[:-1]):
        cv2.polylines(image, [trajectory[c-1:changes[i+1]]], 0, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 3)
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

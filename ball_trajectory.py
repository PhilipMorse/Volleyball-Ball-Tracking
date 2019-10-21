import cv2
import csv
import numpy as np

FILENAME = "vid1-1571401605"


#TODO: create multiple trajectories if certain delay in frames in between
#TODO: identify hit start frame


def separate_trajectories(traj):
    unique_traj = []
    vectors = []
    for i in range(len(traj)-4):
        vectors.append([])

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
    traj = clean_trajectories(traj)
    return traj


if __name__ == "__main__":
    image = cv2.imread("output/" + FILENAME + ".png")

    trajectory = load_trajectories()
    cv2.polylines(image, [trajectory], 0, (0, 0, 255), 3)
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

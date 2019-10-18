import cv2
import csv
import numpy as np

FILENAME = "vid2-1571432796"


#TODO: create multiple trajectories if certain delay in frames in between
#TODO: identify hit start frame
def clean_trajectories(traj):
    output = [[traj[0][0],traj[0][1]]]

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

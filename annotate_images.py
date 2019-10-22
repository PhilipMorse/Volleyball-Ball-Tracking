import glob, os, cv2, csv

DATA_FOLDER = "mikasa"

os.chdir(os.path.join("data",DATA_FOLDER))
print("Controls:")
print("[1] if image shows ball")
print("[0] if image does not show ball")
print("[esc] to exit")
print("Any other key to pass")
with open("annotated_data/annotations.csv", 'a') as annotations:
    csv_writer = csv.writer(annotations, delimiter=',', lineterminator='\n')
    for file in glob.glob("*.png"):
        img = cv2.imread(file)
        img_resize = cv2.resize(img, (0, 0), fx=5, fy=5)
        cv2.imshow("Img", img_resize)
        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == 49:
            cv2.imwrite("annotated_data/" + file, img)
            csv_writer.writerow([file, 1])
            os.remove(file)
        elif k == 48:
            cv2.imwrite("annotated_data/" + file, img)
            csv_writer.writerow([file, 0])
            os.remove(file)

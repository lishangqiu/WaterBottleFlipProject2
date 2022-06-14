import cv2

FILE_NAME = "100mL3"
FILE_PATH = f"./Videos/{FILE_NAME}.MOV"

vid = cv2.VideoCapture(FILE_PATH)
i = 0
while vid.isOpened():
    ret, frame = vid.read()
    if i < 240:
        i += 1
        continue

    cv2.putText(frame, str(i), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imshow("window", cv2.resize(frame, (1280, 720)))
    cv2.waitKey(0)
    i += 1




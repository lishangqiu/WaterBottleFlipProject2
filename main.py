import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter

FILE_NAME = "100mL1"

#LOWER_BLUE_BOUND = np.array([0, 100, 100])
LOWER_BLUE_BOUND = np.array([0, 80, 100])
UPPER_BLUE_BOUND = np.array([179, 255, 255])

START_END = {"100mL1": (53, 197), "100mL2": (87, 240), "150mL1": (37, 209), "150mL2": (40, 215), "200mL1": (68, 230), "200mL2": (106, 240)}
FLIP_FRAME = {"100mL1": 70, "100mL2": 110, "200mL1": 86, "200mL2": 124, "150mL1": 60, "150mL2": 67}

START_FRAME = START_END[FILE_NAME][0]
END_FRAME = START_END[FILE_NAME][1]

DISPLAY = True

VIDEO_PATH = f"./Videos/{FILE_NAME}.MOV"


def diff_angle(angle1, angle2):
    angle_dif = angle1 - angle2
    return (angle_dif + 180) % 360 - 180


def dist_points(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def interpolate(angle1, angle2):
    if angle2 - angle1  < -310:
        print("hi")
        angle_sum = (360 - angle1) + angle2
        return angle1 + (angle_sum / 2)
    return (angle1 + angle2) / 2


vid = cv2.VideoCapture(VIDEO_PATH)


def nothing(x):
    pass
    # Create a window


cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

i = 0
angles = []
pointA_locs = []
pointB_locs = []

contours = []

pbar = tqdm(total=END_FRAME - START_FRAME)
while vid.isOpened():
    i += 1
    ret, frame = vid.read()
    image = cv2.resize(frame, (1280, 720))

    if not ret or i >= END_FRAME:
        print(END_FRAME)
        break

    if i < START_FRAME:
        continue

    if not DISPLAY:
        pbar.update(1)

    # while (1):
    #     # Get current positions of all trackbars
    #     hMin = cv2.getTrackbarPos('HMin', 'image')
    #     sMin = cv2.getTrackbarPos('SMin', 'image')
    #     vMin = cv2.getTrackbarPos('VMin', 'image')
    #     hMax = cv2.getTrackbarPos('HMax', 'image')
    #     sMax = cv2.getTrackbarPos('SMax', 'image')
    #     vMax = cv2.getTrackbarPos('VMax', 'image')
    #
    #     # Set minimum and maximum HSV values to display
    #     lower = np.array([hMin, sMin, vMin])
    #     upper = np.array([hMax, sMax, vMax])
    #
    #     # Convert to HSV format and color threshold
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(hsv, lower, upper)
    #     result = cv2.bitwise_and(image, image, mask=mask)
    #
    #     # Print if there is a change in HSV value
    #     if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
    #         print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
    #         hMin, sMin, vMin, hMax, sMax, vMax))
    #         phMin = hMin
    #         psMin = sMin
    #         pvMin = vMin
    #         phMax = hMax
    #         psMax = sMax
    #         pvMax = vMax
    #
    #     # Display result image
    #     cv2.imshow('image', result)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE_BOUND, UPPER_BLUE_BOUND)
    mask[:, :200] = 0
    dilation = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    processed = cv2.bitwise_and(frame, frame, mask=mask)
    mask = cv2.medianBlur(mask, 11)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        pts = []

        for c in contours:
            if cv2.contourArea(c) < 500:
                break
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            pts.append((cX, cY))
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

        if len(pointA_locs) != 0:
            # Finds the closest point to A and adds to list, same for B(Extremely fucking bad code but it's late)
            pts = sorted(pts, key=lambda pt: dist_points(pt, pointA_locs[-1]))
            pointA_locs.append(pts[0])
            pts.remove(pts[0])

            pts = sorted(pts, key=lambda pt: dist_points(pt, pointB_locs[-1]))
            pointB_locs.append(pts[0])
        else:
            pts = sorted(pts, key=lambda pt: pt[1], reverse=True)
            pointA_locs.append(pts[0])
            pointB_locs.append(pts[1])

        cv2.circle(frame, (pointA_locs[-1][0], pointA_locs[-1][1]), 7, (0, 255, 0), -1)
        cv2.circle(frame, (pointB_locs[-1][0], pointB_locs[-1][1]), 7, (255, 0, 0), -1)
        cv2.line(frame, (pointA_locs[-1][0], pointA_locs[-1][1]), (pointB_locs[-1][0], pointB_locs[-1][1]),
                 (255, 0, 255), thickness=3)
        angle = math.degrees(
            math.atan2(pointA_locs[-1][1] - pointB_locs[-1][1], pointA_locs[-1][0] - pointB_locs[-1][0])) + 90
        if len(angles) == 0 or (angle % 360) != angles[-1]:
            angles.append(angle % 360)
        else:
            angles.append(None)
        cv2.putText(frame, "Angle: " + str(round(angle % 360, 1)), (1400, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
    else:
        print("hi")
    if not DISPLAY:
        continue
    cv2.putText(frame, str(i), (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
    cv2.imshow("window", cv2.resize(frame, (1280, 720)))
    cv2.waitKey(0)
cv2.destroyAllWindows()
pbar.close()
vid.release()

print(angles)

# interpolate
interpolated_angles = []
to_add = 0
for i in range(len(angles)):
    angle = angles[i]
    if angles[i] is None:
        if i == len(angles) - 1:
            angle = angles[i - 1] + (angles[i - 1] - angles[i - 2])
        else:
            angle = interpolate(angles[i - 1], angles[i + 1])

    interpolated_angles.append(angle)

angles = interpolated_angles

new_angles = []
for i in range(len(angles)):
    if angles[i] - angles[i - 1] < -310:
        to_add += 360
    new_angles.append(angles[i] + to_add)
angles = new_angles
filtered_angles = savgol_filter(angles, 31, 3)

print(angles)
print(filtered_angles)

angular_diffs = []
for i in range(len(filtered_angles)):
    if i == 0:
        continue
    diff = diff_angle(filtered_angles[i], filtered_angles[i - 1])
    angular_diffs.append(diff * 30)  # assume 30 fps

plt.ion()

replay_vid = cv2.VideoCapture(VIDEO_PATH)
for i in range(START_FRAME - 1):
    _ = replay_vid.read()

#plt.plot(angles)
plt.plot(savgol_filter(angular_diffs, 31, 3))
plt.ylabel("Angular Velocity (degrees per seconds)")
plt.xlabel(f"Frame #")
plt.title(FILE_NAME)

#line = plt.axvline(0, color='purple')
flip_frame_line = plt.axvline(FLIP_FRAME[FILE_NAME]-START_FRAME, linestyle="--", color="black")


for i in range(len(pointA_locs)):
    ret, frame = replay_vid.read()
    if not ret:
        break

    cv2.circle(frame, (pointA_locs[i][0], pointA_locs[i][1]), 7, (0, 255, 0), -1)
    cv2.circle(frame, (pointB_locs[i][0], pointB_locs[i][1]), 7, (255, 0, 0), -1)

    cv2.line(frame, (pointA_locs[i][0], pointA_locs[i][1]), (pointB_locs[i][0], pointB_locs[i][1]),
             (255, 0, 255), thickness=3)
    cv2.putText(frame, str(i), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    cv2.imshow('frame', cv2.resize(frame, (1280, 720)))
    cv2.waitKey(200)
    #line.set_xdata(i - 1)
    plt.pause(0.0001)

cv2.destroyAllWindows()
plt.show(block=True)
input()

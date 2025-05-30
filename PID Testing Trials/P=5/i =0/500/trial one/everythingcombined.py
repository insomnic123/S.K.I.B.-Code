import cv2
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

### --- CONFIG --- ###
video_path = "C:\\Users\\qazia\\Desktop\\S.K.I.B. Code\\PID Testing Trials\\P=5\\i =0\\500\\trial one\\trialone.mp4"
tape_img_path = "C:\\Users\\qazia\\Desktop\\S.K.I.B. Code\\PID Testing Trials\\TapeStillEdited.png"
output_image_red = 'path_red.png'
output_image_black = 'path_black.png'
deviation_csv = 'deviation_data.csv'
deviation_output_image = 'deviation_output.png'

### --- PART 1: Extract bot path from video --- ###
min_contour_area = 50
min_circularity = 0.71
positions = []

cap = cv2.VideoCapture(video_path)
frame_width, frame_height = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_width is None or frame_height is None:
        frame_height, frame_width = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([ 5, 150, 140])
    upper_orange = np.array([ 20, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if area > min_contour_area and circularity > min_circularity:
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            positions.append((int(x), int(y)))

            cv2.circle(frame, (int(x), int(y)), min_contour_area, (0, 255, 0), 2)  # Green circle with radius 10

            break

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


cap.release()

### --- PART 2: Extract and align tape path (skeleton) --- ###
def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

tape_img = cv2.imread(tape_img_path)
resized_tape = cv2.resize(tape_img, (frame_width, frame_height))
gray = cv2.cvtColor(resized_tape, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
_, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

tape_points = []
for cnt in filtered_contours:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    skeleton = skeletonize(mask)
    ys, xs = np.where(skeleton == 255)
    tape_points += list(zip(xs, ys))

tape_points = sorted(tape_points, key=lambda p: p[1])  # top-to-bottom
# tape_points.reverse()
tape_offset = 0
tape_points = tape_points[tape_offset::]

if tape_points and positions:
    dx = positions[0][0] - tape_points[0][0]
    dy = positions[0][1] - tape_points[0][1]
    aligned_tape_points = [(x + dx, y + dy) for (x, y) in tape_points]
else:
    aligned_tape_points = tape_points

### --- PART 3: Draw both paths --- ###
canvas1 = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
canvas2 = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

for i in range(len(aligned_tape_points) - 1):
    cv2.line(canvas2, aligned_tape_points[i], aligned_tape_points[i+1], (0, 0, 0), 2)
for i in range(1, len(positions)):
    cv2.line(canvas1, positions[i-1], positions[i], (0, 0, 255), 2)

cv2.imwrite(output_image_red, canvas1)
cv2.imwrite(output_image_black, canvas2)

### --- PART 4: Deviation Analysis --- ###
def interpolate_path(points, num_points=2000):
    # Interpolate along the path to fill gaps using linear interpolation
    from scipy.interpolate import interp1d

    points = sorted(points, key=lambda p: p[1])  # sort by Y
    xs, ys = zip(*points)
    t = np.linspace(0, 1, len(xs))
    t_interp = np.linspace(0, 1, num_points)
    fx = interp1d(t, xs, kind='linear', fill_value="extrapolate")
    fy = interp1d(t, ys, kind='linear', fill_value="extrapolate")
    interpolated = np.stack([fx(t_interp), fy(t_interp)], axis=-1)
    return interpolated.astype(np.int32)

def find_line_deviation(red_path, black_path, csv_output_path, output_image_path):
    image_red = cv2.imread(red_path)
    image_black = cv2.imread(black_path)
    hsv_red = cv2.cvtColor(image_red, cv2.COLOR_BGR2HSV)
    hsv_black = cv2.cvtColor(image_black, cv2.COLOR_BGR2HSV)

    # Detect red
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_red, lower_red1, upper_red1) | cv2.inRange(hsv_red, lower_red2, upper_red2)

    # Detect black
    gray = cv2.cvtColor(image_black, cv2.COLOR_BGR2GRAY)
    _, black_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    #black_only_mask = cv2.bitwise_and(black_thresh, cv2.bitwise_not(red_mask)) DO NOT USE!!!
    black_only_mask = black_thresh

    #cv2.imshow("Black Mask", black_thresh)
    #cv2.waitKey(0)

    contours_black, _ = cv2.findContours(black_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    
    if not contours_black or not contours_red:
        raise ValueError("Could not detect both lines.")

    black_pts_raw = max(contours_black, key=cv2.contourArea).reshape(-1, 2)
    red_pts = max(contours_red, key=cv2.contourArea).reshape(-1, 2)

    # Fill gaps in black path
    black_pts_interp = interpolate_path(black_pts_raw, num_points=3000)
    
    lx = []
    ly = []
    for bruh in black_pts_interp:
        lx.append(float(bruh[0]))
        ly.append(float(bruh[1]))

    lxr = []
    lyr = []
    for bruh2 in red_pts:
        lxr.append(float(bruh2[0]))
        lyr.append(float(bruh2[1]))

    point_dict_black = {}
    for i in range(len(ly)):
        all_x = []
        for j in range(len(lx)):
            if ly[j] == ly[i]:
                all_x.append(lx[j])
        point_dict_black[ly[i]] = max(all_x)

    point_dict_red = {}
    for i in range(len(lyr)):
        all_x = []
        for j in range(len(lxr)):
            if lyr[j] == lyr[i]:
                all_x.append(lxr[j])
        point_dict_red[lyr[i]] = max(all_x)

    with open("deviation_data.csv", "w") as f:
        f.write("Red_X,Red_Y,Black_X,Black_Y,Deviation_X\n")

        black_y = list(point_dict_black.keys())
        black_x = list(point_dict_black.values())
        red_y = list(point_dict_red.keys())
        red_x = list(point_dict_red.values())

        for i in range(len(black_y)):
            if i < len(red_y):
                f.write(f"{red_x[i]},{red_y[i]},{black_x[i]},{black_y[i]},{abs(black_x[i]-red_x[i])}\n")

    fig, ax = plt.subplots()
    ax.scatter(list(point_dict_black.keys()), list(point_dict_black.values()), s = 0.1)
    ax.scatter(list(point_dict_red.keys()), list(point_dict_red.values()), s = 0.1)
    ax.set(xlim=(0, 1500), xticks=np.arange(0, 500, 100),
       ylim=(0, 1500), yticks=np.arange(0, 500, 100))
    plt.savefig("sigma.png")


def plot_deviation_area(csv_path):
    red_x, red_y, black_x, black_y = [], [], [], []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            red_x.append(float(row['Red_X']))
            red_y.append(float(row['Red_Y']))
            black_x.append(float(row['Black_X']))
            black_y.append(float(row['Black_Y']))

    # Sort points by red_y (along-path direction)
    combined = sorted(zip(red_y, red_x, black_x, black_y), key=lambda p: p[0])
    red_y_sorted, red_x_sorted, black_x_sorted, black_y_sorted = zip(*combined)

    plt.figure(figsize=(10, 6))

    # image_width = frame_width
    # image_height = frame_height

    # red_x_flipped = [image_width - x for x in red_x_sorted]
    # red_y_flipped = [image_height - y for y in red_y_sorted]

    # plt.plot(red_y_flipped, red_x_flipped, label='Red Path (Bot)', color='red')

    plt.plot(red_y_sorted, red_x_sorted, label='Red Path (Bot)', color='red')
    plt.plot(black_y_sorted, black_x_sorted, label='Black Path (Tape)', color='black')
    plt.fill_between(red_y_sorted, red_x_sorted, black_x_sorted, color='purple', alpha=0.3, label='Deviation Area')
    plt.xlabel('Y-coordinate (along path)')
    plt.ylabel('X-coordinate (position)')
    plt.title('Deviation Between Bot Path and Tape Path')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("deviation_plot.png")
    plt.show()

### --- Run Analysis --- ###
find_line_deviation(output_image_red, output_image_black, deviation_csv, deviation_output_image)
plot_deviation_area(deviation_csv)

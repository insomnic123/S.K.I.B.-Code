import cv2
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

### --- CONFIG --- ###
video_path = 'trialone.mov'
tape_img_path = 'TapeStill.png'
combined_output_image = 'aligned_combined_path.png'
deviation_csv = 'deviation_data.csv'
deviation_output_image = 'deviation_output.png'

### --- PART 1: Extract bot path from video --- ###
min_contour_area = 200
min_circularity = 0.4
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
    lower_orange = np.array([0, 120, 120])
    upper_orange = np.array([20, 255, 255])
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
            break

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
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

tape_points = []
for cnt in filtered_contours:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    skeleton = skeletonize(mask)
    ys, xs = np.where(skeleton == 255)
    tape_points += list(zip(xs, ys))

tape_points = sorted(tape_points, key=lambda p: p[1])  # top-to-bottom

if tape_points and positions:
    dx = positions[0][0] - tape_points[0][0]
    dy = positions[0][1] - tape_points[0][1]
    aligned_tape_points = [(x + dx, y + dy) for (x, y) in tape_points]
else:
    aligned_tape_points = tape_points

### --- PART 3: Draw both paths --- ###
canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

for i in range(1, len(positions)):
    cv2.line(canvas, positions[i-1], positions[i], (0, 0, 255), 2)
for i in range(len(aligned_tape_points) - 1):
    cv2.line(canvas, aligned_tape_points[i], aligned_tape_points[i+1], (0, 0, 0), 2)

cv2.imwrite(combined_output_image, canvas)

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

def find_line_deviation(image_path, csv_output_path, output_image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect red
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Detect black
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, black_thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    black_only_mask = cv2.bitwise_and(black_thresh, cv2.bitwise_not(red_mask))

    contours_black, _ = cv2.findContours(black_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours_black or not contours_red:
        raise ValueError("Could not detect both lines.")

    black_pts_raw = max(contours_black, key=cv2.contourArea).reshape(-1, 2)
    red_pts = max(contours_red, key=cv2.contourArea).reshape(-1, 2)

    black_pts_interp = interpolate_path(black_pts_raw, num_points=3000)
    tree = cKDTree(black_pts_interp)

    results = []
    zero_count = 0
    dists, indices = tree.query(red_pts, distance_upper_bound=250)

    for i, dist in enumerate(dists):
        red_pt = red_pts[i]
        if np.isinf(dist):
            closest_pt = red_pt
            deviation = 0
            zero_count += 1
        else:
            closest_pt = black_pts_interp[indices[i]]
            deviation = round(dist, 2)
        results.append({
            "Red_X": red_pt[0],
            "Red_Y": red_pt[1],
            "Closest_Black_X": closest_pt[0],
            "Closest_Black_Y": closest_pt[1],
            "Deviation_Pixels": deviation
        })

    avg_dev = np.mean([r["Deviation_Pixels"] for r in results])
    print(f"[✓] Red-to-Black avg deviation: {avg_dev:.2f} px")
    print(f"[i] {zero_count} red points assumed overlapping (deviation=0)")

    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Red_X", "Red_Y", "Closest_Black_X", "Closest_Black_Y", "Deviation_Pixels"])
        for r in results:
            writer.writerow([r["Red_X"], r["Red_Y"], r["Closest_Black_X"], r["Closest_Black_Y"], r["Deviation_Pixels"]])

    # Draw output
    output = image.copy()
    for r in results:
        red_pt = (int(r["Red_X"]), int(r["Red_Y"]))
        closest_pt = (int(r["Closest_Black_X"]), int(r["Closest_Black_Y"]))
        if r["Deviation_Pixels"] > 0:
            cv2.line(output, red_pt, closest_pt, (255, 0, 255), 1)
        else:
            cv2.circle(output, red_pt, 2, (0, 255, 0), -1)

    cv2.drawContours(output, [black_pts_raw], -1, (0, 255, 0), 2)
    cv2.drawContours(output, [red_pts], -1, (0, 0, 255), 2)
    cv2.imwrite(output_image_path, output)

def plot_deviation_vs_x(csv_path):
    red_x, deviation = [], []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            red_x.append(float(row['Red_X']))
            deviation.append(float(row['Deviation_Pixels']))

    # Sort by red_x
    sorted_data = sorted(zip(red_x, deviation))
    red_x_sorted, deviation_sorted = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.plot(red_x_sorted, deviation_sorted, color='blue', label='Deviation (px)')
    plt.xlabel('X-coordinate (horizontal position)')
    plt.ylabel('Deviation (pixels)')
    plt.title('Deviation from Tape Path vs. X-Coordinate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("deviation_vs_x.png")
    plt.show()


def plot_deviation_area(csv_path):
    red_x, red_y, black_x, black_y = [], [], [], []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            red_x.append(float(row['Red_X']))
            red_y.append(float(row['Red_Y']))
            black_x.append(float(row['Closest_Black_X']))
            black_y.append(float(row['Closest_Black_Y']))

    # Sort points by red_y (along-path direction)
    combined = sorted(zip(red_y, red_x, black_x, black_y), key=lambda p: p[0])
    red_y_sorted, red_x_sorted, black_x_sorted, black_y_sorted = zip(*combined)

    plt.figure(figsize=(10, 6))
    plt.plot(red_y_sorted, red_x_sorted, label='Red Path (Bot)', color='red')
    plt.plot(red_y_sorted, black_x_sorted, label='Black Path (Tape)', color='black')
    plt.fill_between(red_y_sorted, red_x_sorted, black_x_sorted, color='purple', alpha=0.3, label='Deviation Area')
    plt.xlabel('Y-coordinate (along path)')
    plt.ylabel('X-coordinate (position)')
    plt.title('Deviation Between Bot Path and Tape Path')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("deviation_plot.png")
    plt.show()

def plot_deviation_area_unsorted(csv_path):
    red_x, red_y, black_x, black_y = [], [], [], []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            red_x.append(float(row['Red_X']))
            red_y.append(float(row['Red_Y']))
            black_x.append(float(row['Closest_Black_X']))
            black_y.append(float(row['Closest_Black_Y']))

    plt.figure(figsize=(10, 6))
    plt.plot(red_x, red_y, label='Red Path (Bot)', color='red')
    plt.plot(black_x, black_y, label='Black Path (Tape)', color='black')
    plt.fill_betweenx(red_y, red_x, black_x, color='purple', alpha=0.3, label='Deviation Area')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Deviation Between Bot Path and Tape Path (Unsorted)')
    plt.legend()
    plt.gca().invert_yaxis()  # Optional: for image-style origin
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("deviation_plot_unsorted.png")
    plt.show()


### --- Run Analysis --- ###
find_line_deviation(combined_output_image, deviation_csv, deviation_output_image)
plot_deviation_area(deviation_csv)
plot_deviation_area_unsorted(deviation_csv)
plot_deviation_vs_x(deviation_csv)
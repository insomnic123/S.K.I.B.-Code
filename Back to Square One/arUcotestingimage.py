import cv2
import cv2.aruco as aruco

# Load image
image = cv2.imread("C:\\Users\\qazia\\Desktop\\S.K.I.B. Code\\Back to Square One\\skib.jpg")  # Change this to your image path

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load 4x4_100 dictionary (contains ID 50)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

# Create the detector
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Detect the markers
corners, ids, rejected = detector.detectMarkers(gray)

# If ID 50 is detected, draw a blue square
if ids is not None:
    for i, marker_id in enumerate(ids):
        if marker_id[0] == 50:
            pts = corners[i][0].astype(int)
            for j in range(4):
                cv2.line(image, tuple(pts[j]), tuple(pts[(j + 1) % 4]), (255, 0, 0), 10)

# Show result
cv2.imwrite("ArUco Marker.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

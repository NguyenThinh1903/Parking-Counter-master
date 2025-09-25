# File: debug_detector.py
import cv2
import numpy as np
from src.slot_detector import SlotDetector
import yaml

# Tải cấu hình
with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

video_source = config['video_source']
detector = SlotDetector(config)

# Đọc một khung hình từ video
cap = cv2.VideoCapture(video_source)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Không thể đọc video")
    exit()

# 1. Debug Canny Edge
edges = detector._preprocess_frame_for_lines(frame)
cv2.imshow("1. Canny Edges", edges)
cv2.waitKey(0)

# 2. Debug Hough Lines
lines = detector._detect_lines(edges)
frame_with_lines = frame.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("2. All Hough Lines", frame_with_lines)
cv2.waitKey(0)

# 3. Debug Classified and Merged Lines
vertical, horizontal = detector._classify_lines(lines)
vertical_merged = detector._merge_lines(vertical, 'vertical')
horizontal_merged = detector._merge_lines(horizontal, 'horizontal')

frame_with_merged_lines = frame.copy()
for x1, y1, x2, y2 in vertical_merged:
    cv2.line(frame_with_merged_lines, (x1, y1), (x2, y2), (255, 0, 0), 2) # Dọc = Xanh dương
for x1, y1, x2, y2 in horizontal_merged:
    cv2.line(frame_with_merged_lines, (x1, y1), (x2, y2), (0, 0, 255), 2) # Ngang = Đỏ
cv2.imshow("3. Merged Vertical (Blue) and Horizontal (Red) Lines", frame_with_merged_lines)
cv2.waitKey(0)

# 4. Debug Final Slots
slots = detector._find_slots_from_intersections(vertical_merged, horizontal_merged)
frame_with_slots = frame.copy()
if slots:
    slots_filtered = detector._non_max_suppression(np.array(slots), 0.3)
    for (x1, y1, x2, y2) in slots_filtered:
        cv2.rectangle(frame_with_slots, (x1, y1), (x2, y2), (0, 255, 255), 2)
cv2.imshow("4. Final Detected Slots", frame_with_slots)
cv2.waitKey(0)

cv2.destroyAllWindows()
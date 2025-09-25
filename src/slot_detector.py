import cv2
import numpy as np
import json
import os
from itertools import combinations

class SlotDetector:
    """
    Lớp thực hiện phát hiện tự động các ô đỗ xe từ khung hình video.
    Sử dụng phương pháp phát hiện cạnh, biến đổi Hough để tìm đường thẳng,
    và logic nhóm các đường thẳng để suy ra vị trí các ô.
    """
    
    def __init__(self, config):
        """
        Khởi tạo SlotDetector với các tham số từ cấu hình.
        
        Args:
            config: Cấu hình chứa các tham số phát hiện
        """
        self.detection_params = config.get('detection_params', {})
        # Tinh chỉnh các tham số này trong config.yaml để phù hợp với video của bạn
        self.canny_low_thresh = self.detection_params.get('canny_low_thresh', 50)
        self.canny_high_thresh = self.detection_params.get('canny_high_thresh', 150)
        self.hough_rho = self.detection_params.get('hough_rho', 1)
        self.hough_theta_res = self.detection_params.get('hough_theta_res', np.pi / 180)
        self.hough_threshold = self.detection_params.get('hough_threshold', 40) # Tăng nhẹ threshold
        self.hough_min_line_length = self.detection_params.get('hough_min_line_length', 30) # Tăng nhẹ
        self.hough_max_line_gap = self.detection_params.get('hough_max_line_gap', 5)
        
        # Các tham số cho việc nhóm và tạo ô
        self.slot_width_min = self.detection_params.get('slot_width_min', 25)
        self.slot_width_max = self.detection_params.get('slot_width_max', 60)
        self.slot_height_min = self.detection_params.get('slot_height_min', 50)
        self.slot_height_max = self.detection_params.get('slot_height_max', 120)
        
    def detect(self, video_source):
        """
        Phát hiện các ô đỗ xe từ video đầu vào.
        
        Args:
            video_source: Đường dẫn đến video
            
        Returns:
            List các tọa độ ô đỗ xe dưới dạng [x1, y1, x2, y2]
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Không thể mở video '{video_source}'")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Không thể đọc khung hình từ video '{video_source}'")
        
        edges = self._preprocess_frame_for_lines(frame)
        lines = self._detect_lines(edges)
        
        if lines is None:
            print("[WARNING] Không tìm thấy đường thẳng nào. Hãy thử giảm 'hough_threshold' trong config.")
            return []
            
        vertical_lines, horizontal_lines = self._classify_lines(lines)
        
        # Hợp nhất các đường thẳng gần nhau để giảm nhiễu
        vertical_lines = self._merge_lines(vertical_lines, 'vertical')
        horizontal_lines = self._merge_lines(horizontal_lines, 'horizontal')

        # === THAY ĐỔI QUAN TRỌNG Ở ĐÂY ===
        parking_slots = self._find_slots_from_intersections(vertical_lines, horizontal_lines)
        
        if not parking_slots:
            print("[WARNING] Không tìm thấy ô nào từ giao điểm. Quay lại logic cũ đơn giản hơn để thử...")
            # Nếu logic mới không hoạt động, có thể thử lại logic cũ (dù ít hiệu quả hơn)
            # Hoặc đơn giản là báo lỗi và yêu cầu tinh chỉnh tham số
            # parking_slots = self._find_slots_from_vertical_pairs(vertical_lines)

        # Lọc bỏ các ô trùng lặp
        if parking_slots:
            parking_slots = self._non_max_suppression(np.array(parking_slots), 0.3)
        
        print(f"[*] Đã phát hiện được {len(parking_slots)} ô đỗ xe sau khi lọc.")
        
        return parking_slots.tolist() if isinstance(parking_slots, np.ndarray) else parking_slots

    def _preprocess_frame_for_lines(self, frame):
        """Tiền xử lý ảnh để phát hiện cạnh."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low_thresh, self.canny_high_thresh)
        return edges

    def _detect_lines(self, edges):
        """Sử dụng Hough Transform để phát hiện đoạn thẳng."""
        lines = cv2.HoughLinesP(
            edges, self.hough_rho, self.hough_theta_res, self.hough_threshold,
            minLineLength=self.hough_min_line_length, maxLineGap=self.hough_max_line_gap
        )
        return lines
        
    def _classify_lines(self, lines, angle_thresh=np.pi/6): # Nới lỏng góc một chút
        """Phân loại đường thẳng thành dọc và ngang."""
        vertical = []
        horizontal = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Bỏ qua các đường thẳng quá ngắn
            if np.sqrt((x2-x1)**2 + (y2-y1)**2) < 10: continue

            angle = np.arctan2(y2 - y1, x2 - x1)
            # Ngang: góc gần 0 hoặc PI
            if abs(angle) < angle_thresh or abs(angle - np.pi) < angle_thresh or abs(angle + np.pi) < angle_thresh:
                horizontal.append(line[0])
            # Dọc: góc gần PI/2 hoặc -PI/2
            elif abs(angle - np.pi/2) < angle_thresh or abs(angle + np.pi/2) < angle_thresh:
                vertical.append(line[0])
        return vertical, horizontal
        
    def _merge_lines(self, lines, orientation, dist_thresh=15):
        """Hợp nhất các đoạn thẳng gần nhau và cùng hướng."""
        if not lines:
            return []
        
        # Sắp xếp các đường thẳng dựa trên vị trí của chúng
        if orientation == 'vertical':
            lines = sorted(lines, key=lambda line: line[0])
        else: # horizontal
            lines = sorted(lines, key=lambda line: line[1])

        merged_lines = []
        if not lines:
            return merged_lines

        current_line_group = [lines[0]]
        for i in range(1, len(lines)):
            line = lines[i]
            last_line = current_line_group[-1]
            
            # Tính khoảng cách
            if orientation == 'vertical':
                dist = abs(line[0] - last_line[0])
            else: # horizontal
                dist = abs(line[1] - last_line[1])

            if dist < dist_thresh:
                current_line_group.append(line)
            else:
                # Hợp nhất nhóm cũ và bắt đầu nhóm mới
                x_coords = [l[0] for l in current_line_group] + [l[2] for l in current_line_group]
                y_coords = [l[1] for l in current_line_group] + [l[3] for l in current_line_group]
                
                if orientation == 'vertical':
                    avg_x = int(np.mean(x_coords))
                    merged_lines.append([avg_x, min(y_coords), avg_x, max(y_coords)])
                else:
                    avg_y = int(np.mean(y_coords))
                    merged_lines.append([min(x_coords), avg_y, max(x_coords), avg_y])
                
                current_line_group = [line]

        # Hợp nhất nhóm cuối cùng
        x_coords = [l[0] for l in current_line_group] + [l[2] for l in current_line_group]
        y_coords = [l[1] for l in current_line_group] + [l[3] for l in current_line_group]
        if orientation == 'vertical':
            avg_x = int(np.mean(x_coords))
            merged_lines.append([avg_x, min(y_coords), avg_x, max(y_coords)])
        else:
            avg_y = int(np.mean(y_coords))
            merged_lines.append([min(x_coords), avg_y, max(x_coords), avg_y])
        
        return merged_lines

    # === HÀM MỚI, LOGIC TỐT HƠN ===
    def _find_slots_from_intersections(self, vertical_lines, horizontal_lines):
        """Tìm các ô chữ nhật từ giao điểm của các đường thẳng dọc và ngang."""
        slots = []
        # Lặp qua tất cả các cặp đường thẳng dọc
        for v1, v2 in combinations(vertical_lines, 2):
            # Lặp qua tất cả các cặp đường thẳng ngang
            for h1, h2 in combinations(horizontal_lines, 2):
                # Tọa độ x của hai đường thẳng dọc
                x1, x2 = min(v1[0], v2[0]), max(v1[0], v2[0])
                # Tọa độ y của hai đường thẳng ngang
                y1, y2 = min(h1[1], h2[1]), max(h1[1], h2[1])

                width = x2 - x1
                height = y2 - y1

                # Kiểm tra xem hình chữ nhật có kích thước hợp lý không
                if (self.slot_width_min < width < self.slot_width_max and
                    self.slot_height_min < height < self.slot_height_max):
                    
                    # Kiểm tra xem các đường thẳng có thực sự tạo thành một hình chữ nhật không
                    # (Tức là chúng có "chồng lấn" lên nhau không)
                    y_overlap = max(0, min(v1[3], v2[3]) - max(v1[1], v2[1]))
                    x_overlap = max(0, min(h1[2], h2[2]) - max(h1[0], h2[0]))
                    
                    # Nếu có sự chồng lấn đủ lớn (ví dụ > 50% chiều cao/rộng)
                    if y_overlap > height * 0.5 and x_overlap > width * 0.5:
                        slots.append([x1, y1, x2, y2])

        return slots
        
    def _non_max_suppression(self, boxes, overlapThresh):
        """Lọc bỏ các ô bị trùng lặp nhiều."""
        if len(boxes) == 0:
            return []
        
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            
        return boxes[pick].astype("int")

    def save_slots(self, slots, path):
        """Lưu danh sách ô đỗ xe vào file JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(slots, f)
    
    def load_slots(self, path):
        """Tải danh sách ô đỗ xe từ file JSON."""
        if not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            return json.load(f)
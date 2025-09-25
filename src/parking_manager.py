import cv2
import numpy as np

class ParkingManager:
    """
    Lớp quản lý trạng thái các ô đỗ xe.
    """
    
    def __init__(self, slots, config):
        """
        Khởi tạo ParkingManager.
        
        Args:
            slots: Danh sách các ô đỗ xe dưới dạng [x1, y1, x2, y2]
            config: Cấu hình chứa các tham số quản lý
        """
        self.slots = slots
        self.occupancy_params = config.get('occupancy_params', {})
        self.empty_threshold = self.occupancy_params.get('empty_threshold', 0.25)
        self.stability_threshold = self.occupancy_params.get('stability_threshold', 5)
        self.alpha = self.occupancy_params.get('alpha', 0.5)
        
        # Khởi tạo trạng thái cho các ô đỗ xe
        self.slot_statuses = [{'is_free': True, 'stable_count': 0} for _ in self.slots]
        
    def update_statuses(self, frame):
        """
        Cập nhật trạng thái các ô đỗ xe dựa trên khung hình hiện tại.
        
        Args:
            frame: Khung hình hiện tại từ video
            
        Returns:
            Tuple gồm (số ô trống, tổng số ô, trạng thái từng ô)
        """
        processed_frame = self._preprocess_frame(frame)
        occupied_slots = 0
        
        for i, (xi, yi, xf, yf) in enumerate(self.slots):
            # Cắt phần ảnh tương ứng với ô đỗ xe
            spot_crop = processed_frame[yi:yf, xi:xf]
            
            h, w = spot_crop.shape[:2]
            if h == 0 or w == 0:
                continue
            
            # Tính toán tỷ lệ điểm ảnh không bằng 0 (ô có thể bị chiếm)
            ratio = cv2.countNonZero(spot_crop) / (w * h)
            current_is_free = ratio < self.empty_threshold
            
            # Cập nhật trạng thái với cơ chế ổn định
            status_info = self.slot_statuses[i]
            if current_is_free != status_info['is_free']:
                status_info['stable_count'] += 1
                if status_info['stable_count'] >= self.stability_threshold:
                    status_info['is_free'] = current_is_free
                    status_info['stable_count'] = 0
            else:
                status_info['stable_count'] = 0
            
            # Cập nhật số ô đang được sử dụng
            if not status_info['is_free']:
                occupied_slots += 1
        
        available_slots = len(self.slots) - occupied_slots
        return available_slots, len(self.slots), [status['is_free'] for status in self.slot_statuses]
    
    def _preprocess_frame(self, frame):
        """
        Tiền xử lý khung hình để dễ dàng phát hiện trạng thái ô đỗ xe.
        
        Args:
            frame: Khung hình đầu vào
            
        Returns:
            Ảnh đã xử lý
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        th_frame = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        kernel = np.ones((3, 3), np.uint8)
        processed_frame = cv2.morphologyEx(th_frame, cv2.MORPH_OPEN, kernel, iterations=1)
        return processed_frame
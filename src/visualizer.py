import cv2
import time

class Visualizer:
    def __init__(self, occupancy_params):
        self.alpha = occupancy_params.get('alpha', 0.5)
        self.prev_frame_time = 0
        
    def draw_slots(self, frame, slots, statuses):
        painting_overlay = frame.copy()
        
        for i, (xi, yi, xf, yf) in enumerate(slots):
            is_free = statuses[i]
            color = (0, 255, 0) if is_free else (0, 0, 255)  # Xanh nếu trống, đỏ nếu đầy
            cv2.rectangle(painting_overlay, (xi, yi), (xf, yf), color, -1)
        final_frame = cv2.addWeighted(painting_overlay, self.alpha, frame, 1 - self.alpha, 0)
        
        # **** THAY ĐỔI QUAN TRỌNG: Chỉ trả về frame, không tính toán lại ****
        return final_frame
    
    def draw_ui_panel(self, frame, available_slots, total_slots, fps):
        """
        Vẽ bảng thông tin UI lên khung hình.
        
        Args:
            frame: Khung hình để vẽ lên
            available_slots: Số ô trống
            total_slots: Tổng số ô
            fps: Số khung hình mỗi giây
        """
        panel_height = 100
        panel_color = (0, 0, 0)
        panel_transparency = 0.6

        ui_panel_overlay = frame.copy()
        cv2.rectangle(ui_panel_overlay, (0, 0), (frame.shape[1], panel_height), panel_color, -1)
        cv2.addWeighted(ui_panel_overlay, panel_transparency, frame, 1 - panel_transparency, 0, frame)

        cv2.line(frame, (0, panel_height), (frame.shape[1], panel_height), (0, 255, 0), 2)

        cv2.putText(frame, "PARKING STATUS", (frame.shape[1] // 2 - 150, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        status_text = f"AVAILABLE: {available_slots}/{total_slots}"
        (text_width, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(frame, status_text, (frame.shape[1] - text_width - 20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 220), 2)
    
    def calculate_fps(self):
        """
        Tính toán số khung hình mỗi giây (FPS).
        
        Returns:
            FPS hiện tại
        """
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = new_frame_time
        return fps
import cv2
import yaml
import os
from src.slot_detector import SlotDetector
from src.parking_manager import ParkingManager
from src.visualizer import Visualizer

def load_config(config_path="config/config.yaml"):
    """Tải file cấu hình từ đường dẫn được chỉ định."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def main():
    # Đọc cấu hình
    config = load_config()
    video_source = config['video_source']
    slots_data_path = config['slots_data_path']
    
    # Kiểm tra tọa độ ô đỗ xe
    slot_detector = SlotDetector(config)
    parking_slots = slot_detector.load_slots(slots_data_path)
    
    if not parking_slots:
        print("[*] Không tìm thấy tọa độ ô đỗ xe. Bắt đầu phát hiện tự động...")
        parking_slots = slot_detector.detect(video_source)
        
        if parking_slots:
            slot_detector.save_slots(parking_slots, slots_data_path)
            print(f"[*] Đã tự động phát hiện và lưu {len(parking_slots)} ô đỗ xe.")
        else:
            print("[!] Không phát hiện được ô đỗ xe nào từ video.")
            return
    else:
        print(f"[*] Đã tải {len(parking_slots)} ô đỗ xe từ file đã lưu.")
    
    # Khởi tạo các đối tượng
    parking_manager = ParkingManager(parking_slots, config)
    visualizer = Visualizer(config['occupancy_params'])
    
    # Mở video nguồn
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[!] Lỗi: Không thể mở video '{video_source}'")
        return
    
    print("[*] Bắt đầu chạy hệ thống phát hiện bãi đỗ xe tự động...")
    
    # Khai báo RESIZE_FACTOR ở đây, giá trị phải GIỐNG HỆT trong slot_annotator.py
    RESIZE_FACTOR = 0.7 # <--- THÊM DÒNG NÀY

    # Vòng lặp chính
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # **** THAY ĐỔI QUAN TRỌNG: Luồng dữ liệu được sửa lại ****

        # 1. Manager tính toán và trả về tất cả thông tin trạng thái
        available_slots, total_slots, statuses = parking_manager.update_statuses(frame)
        
        # 2. Visualizer nhận dữ liệu và chỉ vẽ, không tính toán lại
        final_frame = visualizer.draw_slots(frame, parking_slots, statuses)
        
        # 3. Tính toán FPS
        fps = visualizer.calculate_fps()
        
        # 4. Vẽ bảng thông tin UI, sử dụng dữ liệu đã được tính toán ở bước 1
        visualizer.draw_ui_panel(final_frame, available_slots, total_slots, fps)
        
        # Hiển thị khung hình cuối cùng
        # Thay vì hiển thị final_frame, hãy hiển thị frame đã resize
        width = int(final_frame.shape[1] * RESIZE_FACTOR) # <--- THÊM DÒNG NÀY
        height = int(final_frame.shape[0] * RESIZE_FACTOR) # <--- THÊM DÒNG NÀY
        display_frame = cv2.resize(final_frame, (width, height)) # <--- THÊM DÒNG NÀY

        cv2.imshow("Parking Status", display_frame) # <--- SỬA final_frame thành display_frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
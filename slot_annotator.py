import cv2
import json
import yaml

# =================== PHẦN CẦN THAY ĐỔI ===================
# THAY ĐỔI GIÁ TRỊ NÀY ĐỂ CHỈNH KÍCH THƯỚC CỬA SỔ
RESIZE_FACTOR = 0.7 
# ==========================================================

# Cấu hình
CONFIG_PATH = "config/config.yaml"
slots = []
current_slot = None
drawing = False

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def redraw_all_slots():
    """Vẽ lại toàn bộ các ô đã lưu lên frame hiển thị."""
    # Bắt đầu lại với frame sạch
    display_frame[:] = clean_display_frame[:]
    
    # Vẽ lại các ô đã có
    for (x1, y1, x2, y2) in slots:
        disp_x1, disp_y1 = int(x1 * RESIZE_FACTOR), int(y1 * RESIZE_FACTOR)
        disp_x2, disp_y2 = int(x2 * RESIZE_FACTOR), int(y2 * RESIZE_FACTOR)
        cv2.rectangle(display_frame, (disp_x1, disp_y1), (disp_x2, disp_y2), (0, 255, 0), 2)

def mouse_callback(event, x, y, flags, param):
    global current_slot, drawing, slots, display_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_slot = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = display_frame.copy()
            # Vẽ ô hiện tại đang được kéo
            cv2.rectangle(frame_copy, current_slot[0], (x, y), (0, 0, 255), 2)
            cv2.imshow("Slot Annotator", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_slot.append((x, y))
        
        orig_x1 = int(min(current_slot[0][0], current_slot[1][0]) / RESIZE_FACTOR)
        orig_y1 = int(min(current_slot[0][1], current_slot[1][1]) / RESIZE_FACTOR)
        orig_x2 = int(max(current_slot[0][0], current_slot[1][0]) / RESIZE_FACTOR)
        orig_y2 = int(max(current_slot[0][1], current_slot[1][1]) / RESIZE_FACTOR)
        
        slots.append([orig_x1, orig_y1, orig_x2, orig_y2])
        print(f"Added slot: {[orig_x1, orig_y1, orig_x2, orig_y2]}")
        
        # Vẽ lại tất cả các ô để cập nhật
        redraw_all_slots()

if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    video_source = config['video_source']
    slots_data_path = config['slots_data_path']

    cap = cv2.VideoCapture(video_source)
    ret, frame_original = cap.read()
    cap.release()

    if not ret:
        print(f"Cannot read frame from video: {video_source}")
        exit()

    width = int(frame_original.shape[1] * RESIZE_FACTOR)
    height = int(frame_original.shape[0] * RESIZE_FACTOR)
    display_frame = cv2.resize(frame_original, (width, height), interpolation=cv2.INTER_AREA)
    clean_display_frame = display_frame.copy()

    cv2.namedWindow("Slot Annotator")
    cv2.setMouseCallback("Slot Annotator", mouse_callback)

    print("\n" + "="*50)
    print(" HƯỚNG DẪN SỬ DỤNG:")
    print(f"- Cửa sổ đã được chỉnh kích thước về {int(RESIZE_FACTOR * 100)}% so với gốc.")
    print("- Kéo chuột để vẽ một ô đỗ xe.")
    print("\n CÁC PHÍM TẮT:")
    print("- 's': Lưu tất cả các ô và thoát.")
    print("- 'q': Thoát không lưu.")
    print("- 'c': Xóa TẤT CẢ các ô đã vẽ.")
    print("- 'z': Xóa ô VỪA VẼ GẦN NHẤT (Undo).")
    print("="*50 + "\n")

    while True:
        cv2.imshow("Slot Annotator", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            with open(slots_data_path, 'w') as f:
                json.dump(slots, f)
            print(f"Successfully saved {len(slots)} slots to {slots_data_path}")
            break
        elif key == ord('q'):
            break
        elif key == ord('c'):
            slots = []
            redraw_all_slots() # Vẽ lại frame trống
            print("Cleared all slots.")
        
        # === TÍNH NĂNG MỚI: UNDO ===
        elif key == ord('z'):
            if slots:
                removed_slot = slots.pop()
                print(f"Removed last slot: {removed_slot}")
                redraw_all_slots()
            else:
                print("No slots to remove.")

    cv2.destroyAllWindows()
import time
import keyboard
import cv2
import numpy as np
import mss
import pyautogui
import win32api
import win32gui
import win32con
import pygetwindow as gw
from ultralytics import YOLO

model = YOLO("runs/detect/train8/weights/best.pt")

last_bobber_position = None
motion_threshold = 2.2
window_rect = None

def get_window_rect(window_title_substring):
    """Находит и возвращает координаты окна Terraria."""
    global window_rect

    if window_rect:
        return window_rect

    for window in gw.getWindowsWithTitle(''):
        if window_title_substring.lower() in window.title.lower():
            hwnd = window._hWnd
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            window_rect = rect
            return rect
    return None

def capture_window(window_title_substring):
    """Захватывает изображение Terraria."""
    rect = get_window_rect(window_title_substring)
    if rect:
        x1, y1, x2, y2 = rect
        width, height = x2 - x1, y2 - y1

        with mss.mss() as sct:
            monitor = {"top": y1, "left": x1, "width": width, "height": height}
            sct_img = sct.grab(monitor)

            img = np.array(sct_img, dtype=np.uint8)
            if img is None or img.size == 0:
                print("Ошибка: пустой кадр!")
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
    return None

def find_bobber(frame):
    """Обнаружение поплавка и других объектов на кадре с помощью YOLO."""
    results = model(frame, conf=0.3, verbose=False)
    bobber_position = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = model.names[class_id] if class_id in model.names else f"ID {class_id}"

            if class_name.lower() == "bobber":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                bobber_position = ((x1 + x2) // 2, (y1 + y2) // 2)

    return bobber_position, frame

def bobber_motion_found(bobber_position):
    """Проверяет, двигается ли поплавок."""
    global last_bobber_position
    if last_bobber_position is None:
        last_bobber_position = bobber_position
        return False

    motion_detected = np.linalg.norm(np.array(bobber_position) - np.array(last_bobber_position)) >= motion_threshold
    last_bobber_position = bobber_position
    return motion_detected

fishing_active = False
def toggle_fishing():
    """Переключает авто-рыбалку"""
    global fishing_active
    fishing_active = not fishing_active
    print("Авто-рыбалка ВКЛ 🟢" if fishing_active else "Авто-рыбалка ВЫКЛ 🔴")

keyboard.add_hotkey("F1", toggle_fishing)
cv2.namedWindow("(Live)")

def left_click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)  # Задержка для стабильности
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

while True:
    current_frame = capture_window("Terraria")
    if current_frame is None:
        print("Ошибка: не удалось захватить окно Terraria.")
        continue

    bobber_position, current_frame = find_bobber(current_frame)

    if fishing_active and bobber_position and bobber_motion_found(bobber_position):
        print("Поплавок движется! Подсекаем!")
        left_click()
        time.sleep(1)
        left_click()
        time.sleep(2)
        current_frame = capture_window("Terraria")
        bobber_position, current_frame = find_bobber(current_frame)
        last_bobber_position = bobber_position

    cv2.imshow("(Live)", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

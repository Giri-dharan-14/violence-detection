import cv2
from datetime import datetime
import os
from telegram_utils import send_to_telegram

# Helper function to draw a banner on the frame
def draw_banner(frame, violence_detected):
    banner_text = "VIOLENCE DETECTED" if violence_detected else "NORMAL"
    color = (0, 0, 255) if violence_detected else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)
    cv2.putText(frame, banner_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def get_violence_class_id(results):
    """
    Dynamically find the class ID for 'violence' from the model's own name mapping.
    This avoids hardcoding the wrong class ID.
    """
    for cls_id, name in results[0].names.items():
        if name.lower() == "violence":
            return cls_id
    return None  # Not found


def has_violence(results, violence_cls=None):
    """
    Check if violence is detected.
    If violence_cls is None, auto-detect the violence class ID from model names.
    """
    if violence_cls is None:
        violence_cls = get_violence_class_id(results)
        if violence_cls is None:
            print("[WARNING] Could not find 'violence' class in model names. Check your data.yaml.")
            return False

    for box in results[0].boxes:
        if int(box.cls) == violence_cls:
            return True
    return False


def capture_and_send(frame, output_dir, bot_token, chat_id):
    """Capture the frame, save it, and send it to Telegram."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{output_dir}violence_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    print(f"[INFO] Image saved: {image_path}")
    send_to_telegram(image_path, bot_token, chat_id)
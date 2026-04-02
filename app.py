from ultralytics import YOLO
import cv2
import torch
import os
from datetime import datetime
import requests
from helpers import draw_banner, has_violence, get_violence_class_id, capture_and_send

# ── Configuration ─────────────────────────────────────────
MODEL_PATH   = "best.pt"
OUTPUT_DIR   = "outputs/"
CONFIDENCE   = 0.8
TELEGRAM_BOT_TOKEN = "XXXXXXX"
TELEGRAM_CHAT_ID   = "XXXXXXX"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────
print("\nLoading model...")
model = YOLO(MODEL_PATH)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Running on : {'GPU — ' + torch.cuda.get_device_name(0) if DEVICE == 0 else 'CPU'}")
print(f"Model      : {MODEL_PATH}\n")

# ── Print class mapping once at startup so you can verify ─
print("[INFO] Model class mapping:")
for cls_id, name in model.names.items():
    print(f"  Class {cls_id} → {name}")
print()


# ─────────────────────────────────────────────────────────
#  MODE 1 — WEBCAM
# ─────────────────────────────────────────────────────────
def run_webcam():
    print("\n[WEBCAM] Starting... Press Q to quit.\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        results  = model(frame, conf=CONFIDENCE, device=DEVICE, verbose=False)
        violence = has_violence(results)          # ← auto-detects class ID
        annotated = results[0].plot()
        annotated = draw_banner(annotated, violence)

        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, ts,
                    (annotated.shape[1] - 120, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if violence:
            capture_and_send(annotated, OUTPUT_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

        cv2.imshow("Violence Detection — Webcam  [Q: quit  S: screenshot]", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{OUTPUT_DIR}webcam_{ts_file}.jpg"
            cv2.imwrite(path, annotated)
            print(f"Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam session ended.")


# ─────────────────────────────────────────────────────────
#  MODE 2 — VIDEO FILE
# ─────────────────────────────────────────────────────────
def run_video(video_path):
    if not os.path.exists(video_path):
        print(f"ERROR: File not found — {video_path}")
        return

    print(f"\n[VIDEO] Processing: {video_path}\n")

    cap    = cv2.VideoCapture(video_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    filename    = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{OUTPUT_DIR}{filename}_detected.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_count     = 0
    violence_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results  = model(frame, conf=CONFIDENCE, device=DEVICE, verbose=False)

        # ── Dynamically resolve the violence class ID from model names ──
        violence_cls_id = get_violence_class_id(results)
        violence = has_violence(results)   # uses auto-detection internally

        # ── Draw annotations BEFORE any use of 'annotated' ──────────────
        annotated = results[0].plot()
        annotated = draw_banner(annotated, violence)

        if violence:
            violence_frames += 1

            # Only send high-confidence detections, up to 5 times
            high_conf_boxes = [
                box for box in results[0].boxes
                if int(box.cls) == violence_cls_id and box.conf.item() > 0.7
            ]

            if high_conf_boxes and violence_frames <= 5:
                print(f"[ALERT] Frame {frame_count}: violence detected with {len(high_conf_boxes)} high-conf box(es)")
                capture_and_send(annotated, OUTPUT_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            elif not high_conf_boxes:
                print(f"[INFO] Frame {frame_count}: violence class present but below 0.7 confidence threshold")

        # Progress counter
        cv2.putText(annotated,
                    f"{frame_count}/{total}",
                    (width - 160, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out.write(annotated)
        cv2.imshow("Violence Detection — Video  [Q: quit]", annotated)

        if frame_count % 50 == 0:
            pct = round(frame_count / total * 100, 1)
            print(f"  Progress: {frame_count}/{total} frames ({pct}%)")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped early by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    violence_pct = round(violence_frames / max(frame_count, 1) * 100, 1)
    print(f"\nDone.")
    print(f"Total frames      : {frame_count}")
    print(f"Violence frames   : {violence_frames} ({violence_pct}%)")
    print(f"Output saved to   : {output_path}")


# ─────────────────────────────────────────────────────────
#  MODE 3 — IMAGE FILE
# ─────────────────────────────────────────────────────────
def run_image(image_path):
    if not os.path.exists(image_path):
        print(f"ERROR: File not found — {image_path}")
        return

    print(f"\n[IMAGE] Processing: {image_path}\n")

    results  = model(image_path, conf=CONFIDENCE, device=DEVICE)
    violence = has_violence(results)   # auto-detects class ID

    annotated  = results[0].plot()
    annotated  = draw_banner(annotated, violence)

    filename    = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{OUTPUT_DIR}{filename}_detected.jpg"
    cv2.imwrite(output_path, annotated)

    print(f"Result    : {'VIOLENCE DETECTED' if violence else 'NORMAL'}")
    print(f"Saved to  : {output_path}\n")

    print("[INFO] All detections:")
    for box in results[0].boxes:
        cls    = int(box.cls)
        conf   = float(box.conf)
        name   = results[0].names.get(cls, "Unknown")
        coords = [round(x, 1) for x in box.xyxy[0].tolist()]
        print(f"  Class {cls} ({name:15s}) — {round(conf*100,1)}% — box: {coords}")

    print("\n[INFO] Class name mapping from model:")
    for cls_id, name in results[0].names.items():
        print(f"  Class {cls_id} → {name}")

    cv2.imshow("Violence Detection — Image  [any key: close]", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   Violence Detection System")
    print("=" * 50)
    print("  1 — Webcam (live)")
    print("  2 — Video file")
    print("  3 — Image file")
    print("=" * 50)

    choice = input("Select mode (1 / 2 / 3): ").strip()

    if choice == "1":
        run_webcam()
    elif choice == "2":
        path = input("Enter video file path: ").strip().strip('"')
        run_video(path)
    elif choice == "3":
        path = input("Enter image file path: ").strip().strip('"')
        run_image(path)
    else:
        print("Invalid choice. Enter 1, 2, or 3.")

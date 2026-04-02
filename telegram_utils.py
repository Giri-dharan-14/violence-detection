import requests
import threading

def send_to_telegram(image_path, bot_token, chat_id):
    """Send image to Telegram in a background thread so it never blocks the main loop."""
    def _send():
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(image_path, "rb") as photo:
                response = requests.post(
                    url,
                    data={"chat_id": chat_id},
                    files={"photo": photo},
                    timeout=10           # don't hang forever if network is slow
                )
            if response.status_code == 200:
                print(f"[TELEGRAM] Sent: {image_path}")
            else:
                print(f"[TELEGRAM] Failed ({response.status_code}): {image_path}")
        except Exception as e:
            print(f"[TELEGRAM] Error: {e}")

    # Fire-and-forget — main loop continues immediately
    threading.Thread(target=_send, daemon=True).start()
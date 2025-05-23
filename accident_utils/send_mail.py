import yagmail
import os

def send_email_alert(subject, content):
    sender = os.getenv("ALERT_EMAIL")
    password = os.getenv("ALERT_PASSWORD")
    recipient = os.getenv("ALERT_RECIPIENT")

    if not all([sender, password, recipient]):
        print("[Email Error] Missing environment variables.")
        return

    try:
        yag = yagmail.SMTP(user=sender, password=password)
        yag.send(recipient, subject, content)
        print("[Email] Alert sent successfully.")
    except Exception as e:
        print(f"[Email Error] {e}")

# Camera Feeds Directory

# Place your camera feed videos here with the following names:
# - cam1.mp4 - Main Entrance
# - cam2.mp4 - Toll Gate
# - cam3.mp4 - Highway Junction
# - cam4.mp4 - Bridge

# Supported formats:
# - MP4 (recommended)
# - AVI
# - MOV

# Note: If no video is available, the dashboard will show an offline status for that camera.

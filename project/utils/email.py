import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(config, recipient, subject, body, attachment_path=None):
    """
    Upgrades vs original:
      - Attachment handling actually implemented (was a placeholder
        comment before, so attachments were silently dropped).
      - Returns (success, error_message) instead of a bare bool, so
        callers can surface the real failure reason instead of just
        "email failed".
      - This is the single implementation now used everywhere (the
        GUI previously duplicated this logic inline in two separate
        methods with no attachment support in one of them).
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = config["user"]
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        if attachment_path:
            with open(attachment_path, "rb") as file:
                attachment = MIMEBase("application", "octet-stream")
                attachment.set_payload(file.read())

            encoders.encode_base64(attachment)
            attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename=os.path.basename(attachment_path),
            )
            msg.attach(attachment)

        with smtplib.SMTP(config["host"], config["port"], timeout=10) as server:
            server.starttls()
            server.login(config["user"], config["password"])
            server.sendmail(config["user"], recipient, msg.as_string())

        return True, None

    except Exception as e:
        return False, str(e)

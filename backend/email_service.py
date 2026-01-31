"""
Email service for verification and password reset.
Uses SMTP when configured; falls back to console logging in development.
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "noreply@hydronyx.local")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


def _smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


def send_verification_email(to_email: str, token: str, name: str) -> bool:
    """Send email verification link."""
    verify_url = f"{FRONTEND_URL}/verify-email?token={token}"
    subject = "Verify your Hydronyx account"
    body = f"""
Hello {name},

Please verify your email address by clicking the link below:

{verify_url}

This link expires in 24 hours.

If you did not create an account, you can ignore this email.

— Hydronyx Team
"""
    return _send_email(to_email, subject, body)


def send_password_reset_email(to_email: str, token: str) -> bool:
    """Send password reset link."""
    reset_url = f"{FRONTEND_URL}/reset-password?token={token}"
    subject = "Reset your Hydronyx password"
    body = f"""
You requested a password reset for your Hydronyx account.

Click the link below to reset your password:

{reset_url}

This link expires in 1 hour.

If you did not request this, you can safely ignore this email.

— Hydronyx Team
"""
    return _send_email(to_email, subject, body)


def _send_email(to: str, subject: str, body: str) -> bool:
    if _smtp_configured():
        try:
            msg = MIMEMultipart()
            msg["From"] = SMTP_FROM
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body.strip(), "plain"))
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM, to, msg.as_string())
            return True
        except Exception as e:
            print(f"[EMAIL] Failed to send: {e}")
            return False
    else:
        print(f"[EMAIL] (dev mode) To: {to}\nSubject: {subject}\n{body}\n")
        return True

#!/usr/bin/env python3
"""
remote_lock_server.py — Lightweight HTTP server to run on Raspberry Pi 5.
Receives unlock commands from the face recognition app over the network.

Requires: sudo apt install python3-rpi-lgpio -y
Usage: python3 remote_lock_server.py
"""

import http.server
import socketserver
import threading
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LockServer")

PORT = 8000
GPIO_PIN = 18          # BCM Pin connected to the solenoid relay
ACTIVE_HIGH = True     # True if HIGH = unlocked, False if LOW = unlocked
UNLOCK_DURATION = 3.0  # seconds the lock stays open before auto-relocking

# Try importing the GPIO library (rpi-lgpio on Pi 5)
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW if ACTIVE_HIGH else GPIO.HIGH)
    logger.info("GPIO initialized successfully (running in hardware mode).")
except ImportError:
    GPIO = None
    logger.warning("RPi.GPIO / rpi-lgpio not available. Running in SOFTWARE SIMULATION mode.")

# Lock state lock to prevent concurrent triggers overlapping
lock_state_mutex = threading.Lock()
is_unlocked = False

def perform_unlock():
    global is_unlocked
    with lock_state_mutex:
        if is_unlocked:
            logger.info("Unlock requested but door is already unlocked. Resetting relock timer.")
            return
        is_unlocked = True

    logger.info("TRIGGER: Unlocking door solenoid...")
    if GPIO:
        GPIO.output(GPIO_PIN, GPIO.HIGH if ACTIVE_HIGH else GPIO.LOW)

    time.sleep(UNLOCK_DURATION)

    if GPIO:
        GPIO.output(GPIO_PIN, GPIO.LOW if ACTIVE_HIGH else GPIO.HIGH)
    logger.info("TRIGGER: Auto-relocked door solenoid.")

    with lock_state_mutex:
        is_unlocked = False

class LockHTTPHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to log via standard logger
        logger.info(format % args)

    def do_POST(self):
        if self.path == "/unlock":
            # Start unlock cycle in a separate thread to return HTTP response instantly
            threading.Thread(target=perform_unlock, daemon=True).start()
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"status": "success", "message": "unlocked"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            state = "unlocked" if is_unlocked else "locked"
            self.wfile.write(f'{{"status": "{state}"}}'.encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    # Allow address reuse to prevent "Address already in use" errors on restarts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), LockHTTPHandler) as httpd:
        logger.info("Remote Lock Server listening on port %d...", PORT)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            if GPIO:
                GPIO.cleanup()
                logger.info("GPIO cleaned up.")

if __name__ == "__main__":
    run_server()

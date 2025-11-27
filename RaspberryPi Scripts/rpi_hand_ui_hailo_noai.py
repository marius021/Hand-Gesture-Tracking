import cv2
import numpy as np
import time
import pandas as pd
import socket
from logger import DataLogger

####################################################
# CONFIG
####################################################

MAX_COMMANDS = 100
CSV_NAME = "session_no_ai.csv"

TCP_IP = "192.168.88.252"
TCP_PORT = 5005

ROI_X1, ROI_Y1 = 100, 100
ROI_X2, ROI_Y2 = 400, 400

####################################################
# TCP CLIENT
####################################################

class CommandSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1)
            self.sock.connect((self.ip, self.port))
            print("[TCP] Connected.")
        except:
            print("[TCP] Failed to connect.")
            self.sock = None

    def send(self, command):
        if self.sock is None:
            self.connect()
            return False

        try:
            self.sock.sendall((command + "\n").encode("utf-8"))
            return True
        except:
            print("[TCP] Lost connection — reconnecting...")
            self.sock = None
            return False


####################################################
# MAIN LOOP (NO AI)
####################################################

def main():
    print("===== STARTING NON-AI SYSTEM =====")

    sender = CommandSender(TCP_IP, TCP_PORT)
    logger = DataLogger(CSV_NAME, pretty_timestamp=True)

    cap = cv2.VideoCapture(0)
    last_cmd = ""
    sent_count = 0

    print("[SYSTEM] Running... Press 'q' to quit.")

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame error.")
            break

        # Region of interest
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        roi_h, roi_w, _ = roi.shape

        #############################
        # BASIC SKIN SEGMENTATION
        #############################

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70])
        upper = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = -1, -1
        command = "No hand"

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)

            if cv2.contourArea(c) > 3000:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if cx < roi_w // 3:
                        command = "Move Left"
                    elif cx > 2 * (roi_w // 3):
                        command = "Move Right"
                    else:
                        command = "Centered"

                cv2.circle(roi, (cx, cy), 7, (0, 255, 255), -1)

        #############################
        # SEND COMMAND
        #############################

        tcp_sent = 0
        tcp_reconnect = 0

        if command != last_cmd:
            ok = sender.send(command)
            tcp_sent = 1
            if not ok:
                tcp_reconnect = 1

            last_cmd = command
            sent_count += 1

            print(f"[CMD] {command} ({sent_count}/{MAX_COMMANDS})")

        #############################
        # STOP CONDITION (100 commands)
        #############################

        if sent_count >= MAX_COMMANDS:
            print("[SYSTEM] Max commands reached.")
            logger.close()
            break

        #############################
        # LOGGING
        #############################

        dt = (time.time() - t0) * 1000
        fps = 1000 / max(dt, 0.0001)

        logger.log(
            fps=fps,
            frame_time_ms=dt,
            hailo_score=0.0,           # no AI
            hailo_valid=0,             # no AI
            mediapipe_points=0,        # no AI
            command=command,
            tcp_sent=tcp_sent,
            tcp_reconnected=tcp_reconnect,
            cx=cx,
            cy=cy
        )

        #############################
        # DISPLAY
        #############################

        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 0), 2)
        cv2.putText(frame, command, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("UI NON-AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.close()
            break

    cap.release()
    cv2.destroyAllWindows()


####################################################
# RUN
####################################################

if __name__ == "__main__":
    main()


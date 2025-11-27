import cv2
import numpy as np
import time
import pandas as pd
import socket
import mediapipe as mp

from logger import DataLogger

####################################################
# CONFIG
####################################################

hailo_enabled = True
MAX_COMMANDS = 100
CSV_NAME = "session_ai.csv"

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
# HAILO OLD API (compatible with HailoRT 4.20)
####################################################

from hailo_platform import VDevice, HEF

def init_hailo():
    print("[Hailo] Initializing (Old API)...")

    device = VDevice()
    hef = HEF("hand_landmark_lite.hef")

    network_groups = hef.get_network_group_names()

    if len(network_groups) == 0:
        raise RuntimeError("No network groups found in HEF.")

    net_name = network_groups[0]
    print(f"[Hailo] Using network group: {net_name}")

    ng = device.configure(hef, net_name)

    input_infos = hef.get_input_vstream_infos(net_name)
    output_infos = hef.get_output_vstream_infos(net_name)

    input_vstream = VDevice.create_input_vstream(device, input_infos[0])
    output_vstreams = [VDevice.create_output_vstream(device, info) for info in output_infos]

    print("[Hailo] READY (Old API).")
    return device, input_vstream, output_vstreams

####################################################
# RUN HAILO
####################################################

def run_hailo(input_vstream, output_vstreams, img_224):
    img_224 = img_224.astype(np.uint8)
    input_vstream.send(img_224)

    outputs = [out.recv() for out in output_vstreams]
    fc1 = outputs[0].flatten()

    score = np.mean(np.abs(fc1))
    return score

####################################################
# MAIN LOOP
####################################################

def main():
    global hailo_enabled
    print("===== STARTING SYSTEM =====")

    sender = CommandSender(TCP_IP, TCP_PORT)
    logger = DataLogger(CSV_NAME, pretty_timestamp=True)

    mp_hands = None

    if hailo_enabled:
        try:
            device, input_vstream, output_vstreams = init_hailo()
        except Exception as e:
            print("[Hailo] ERROR:", e)
            hailo_enabled = False

    cap = cv2.VideoCapture(0)
    last_cmd = ""
    sent_count = 0

    print("[SYSTEM] Running... Press 'q' to quit.")

    while True:
        start_t = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame error.")
            break

        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        roi_h, roi_w, _ = roi.shape

        if mp_hands is None:
            print("[MediaPipe] Initializing...")
            mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[MediaPipe] READY.")

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0,20,70])
        upper = np.array([20,255,255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

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

                    if cx < roi_w//3:
                        command = "Move Left"
                    elif cx > 2*(roi_w//3):
                        command = "Move Right"
                    else:
                        command = "Centered"

        hailo_score = 0.0
        hailo_valid = False

        if hailo_enabled and cx != -1:
            img_224 = cv2.resize(roi, (224,224))
            img_224 = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)

            hailo_score = run_hailo(input_vstream, output_vstreams, img_224)
            hailo_valid = hailo_score > 0.01

            if not hailo_valid:
                command = "No hand"

        mediapipe_points = 0

        mp_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(mp_rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            mediapipe_points = len(lm.landmark)

            for p in lm.landmark:
                lx = int(p.x * roi_w)
                ly = int(p.y * roi_h)
                cv2.circle(roi, (lx, ly), 3, (0,255,0), -1)

        tcp_sent = 0
        tcp_reconn = 0

        if command != last_cmd:
            ok = sender.send(command)
            tcp_sent = 1
            if not ok:
                tcp_reconn = 1

            last_cmd = command
            sent_count += 1

            print(f"[CMD] {command} ({sent_count}/{MAX_COMMANDS})")

        if sent_count >= MAX_COMMANDS:
            print("[SYSTEM] Max commands reached.")
            logger.close()
            break

        dt = (time.time() - start_t)*1000
        fps = 1000/max(dt,0.0001)

        if hailo_enabled and cx != -1:
            print(f"[DEBUG] Hailo used: score={hailo_score:.4f}, valid={hailo_valid}")
        else:
            print(f"[DEBUG] Hailo NOT used (hailo_enabled={hailo_enabled}, cx={cx})")

        logger.log(
            fps=fps,
            frame_time_ms=dt,
            hailo_score=hailo_score,
            hailo_valid=1 if hailo_valid else 0,
            mediapipe_points=mediapipe_points,
            command=command,
            tcp_sent=tcp_sent,
            tcp_reconnected=tcp_reconn,
            cx=cx,
            cy=cy
        )

        cv2.rectangle(frame, (ROI_X1,ROI_Y1),(ROI_X2,ROI_Y2), (255,255,0),2)
        cv2.putText(frame, command, (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if hailo_valid else (0,0,255),2)

        cv2.imshow("UI Final", frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            logger.close()
            break

    cap.release()
    cv2.destroyAllWindows()

####################################################
# RUN APP
####################################################

if __name__ == "__main__":
    main()


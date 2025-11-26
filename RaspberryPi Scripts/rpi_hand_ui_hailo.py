import cv2
import numpy as np
import os
import time
import socket
import mediapipe as mp
import csv

# ---------------------------------------------------
# Config global
# ---------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEF_PATH = os.path.join(SCRIPT_DIR, "models", "hand_landmark_lite.hef")

CAMERA_ID = 0

# ROI fix
ROI_TOP, ROI_BOTTOM = 100, 400
ROI_RIGHT, ROI_LEFT = 100, 400

LEFT_TH_RATIO = 1/3
RIGHT_TH_RATIO = 2/3

HAILO_SCORE_TH = 0.01

# HSV skin model
LOWER_SKIN = np.array([0, 20, 70], dtype=np.uint8)
UPPER_SKIN = np.array([20, 255, 255], dtype=np.uint8)

MODEL_INPUT_SIZE = 224

# Server industrial
ROBOT_HOST = "192.168.0.100"
ROBOT_PORT = 5000


# ---------------------------------------------------
# CSV LOGGER
# ---------------------------------------------------
class DataLogger:
    def __init__(self, filename="gestures_session.csv"):
        self.filename = filename
        file_exists = os.path.exists(self.filename)

        self.file = open(self.filename, "a", newline="")
        self.writer = csv.writer(self.file)

        if not file_exists:
            self.writer.writerow([
                "timestamp",
                "fps",
                "frame_time_ms",
                "hailo_score",
                "hailo_valid",
                "mediapipe_points",
                "command",
                "tcp_sent",
                "tcp_reconnected",
                "cx",
                "cy"
            ])

    def log(self, fps, frame_time_ms, hailo_score, hailo_valid,
            mediapipe_points, command, tcp_sent, tcp_reconnected, cx, cy):

        self.writer.writerow([
            time.time(),
            fps,
            frame_time_ms,
            hailo_score,
            hailo_valid,
            mediapipe_points,
            command,
            tcp_sent,
            tcp_reconnected,
            cx,
            cy
        ])

    def close(self):
        self.file.close()


# ---------------------------------------------------
# TCP Communication
# ---------------------------------------------------

class CommandSender:
    def __init__(self, host=ROBOT_HOST, port=ROBOT_PORT, reconnect_interval=5.0):
        self.host = host
        self.port = port
        self.reconnect_interval = reconnect_interval
        self.sock = None
        self.last_connect_attempt = 0.0

    def _connect(self):
        now = time.time()
        if self.sock is not None:
            return
        if now - self.last_connect_attempt < self.reconnect_interval:
            return

        self.last_connect_attempt = now

        try:
            print(f"[Comm] Connecting to {self.host}:{self.port}...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.host, self.port))
            s.settimeout(None)
            self.sock = s
            print("[Comm] Connected.")
        except Exception as e:
            print(f"[Comm] Connection error: {e}")
            self.sock = None

    def send_command(self, command: str):
        if not command:
            return

        if self.sock is None:
            self._connect()

        if self.sock is None:
            return

        msg = (command.strip() + "\n").encode("utf-8")

        try:
            self.sock.sendall(msg)
            tcp_reconnected = 0
        except Exception:
            tcp_reconnected = 1
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

        return tcp_reconnected

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None


def map_command_to_protocol(cmd: str) -> str:
    mapping = {
        "Move Left": "MOVE_LEFT",
        "Move Right": "MOVE_RIGHT",
        "Centered": "CENTERTERED",
        "No hand": "NO_HAND"
    }
    return mapping.get(cmd, "UNKNOWN")


# ---------------------------------------------------
# Procesare Clasică OpenCV
# ---------------------------------------------------

def process_frame_classic(frame_bgr):
    frame = frame_bgr.copy()
    frame = cv2.flip(frame, 1)

    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_RIGHT:ROI_LEFT]
    roi_vis = roi.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy = None, None
    command = "No hand"

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.drawContours(roi_vis, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(roi_vis, (cx, cy), 5, (0, 0, 255), -1)

            width = roi.shape[1]
            left_th = int(width * LEFT_TH_RATIO)
            right_th = int(width * RIGHT_TH_RATIO)

            if cx < left_th:
                command = "Move Left"
            elif cx > right_th:
                command = "Move Right"
            else:
                command = "Centered"

    cv2.rectangle(frame, (ROI_RIGHT, ROI_TOP), (ROI_LEFT, ROI_BOTTOM), (255, 0, 0), 2)

    return command, frame, roi, roi_vis, mask, (cx, cy)


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    # ---- Camera ----
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Nu pot deschide camera.")
        return

    logger = DataLogger()
    sender = CommandSender()
    last_sent_cmd = None

    hailo_enabled = False
    hailo_score = 0.0
    hailo_valid = False

    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Hailo init
    try:
        from hailo_platform import (
            HEF, VDevice, HailoStreamInterface, ConfigureParams,
            InputVStreams, OutputVStreams,
            InputVStreamParams, OutputVStreamParams,
            FormatType
        )

        if os.path.exists(HEF_PATH):
            target = VDevice()
            hef = HEF(HEF_PATH)
            cfg = ConfigureParams.create_from_hef(hef, HailoStreamInterface.PCIe)
            ng = target.configure(hef, cfg)
            ng = ng[0]
            ng_params = ng.create_params()

            in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

            hailo_enabled = True
            print("[Hailo] Activ.")
    except Exception as e:
        print(f"[Hailo] Error: {e}")
        hailo_enabled = False

    frame_idx = 0

    try:
        if hailo_enabled:

            with ng.activate(ng_params), \
                 InputVStreams(ng, in_params) as in_streams, \
                 OutputVStreams(ng, out_params) as out_streams:

                in_stream = list(in_streams)[0]

                landmark_stream = None
                for s in out_streams:
                    if np.prod(s.shape) == 63:
                        landmark_stream = s

                if landmark_stream is None:
                    hailo_enabled = False

                while True:
                    frame_start = time.time()

                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1

                    command, frame_vis, roi, roi_vis, mask, (cx, cy) = process_frame_classic(frame)

                    # Hailo
                    if roi is not None and cx is not None and frame_idx % 3 == 0:
                        resized = cv2.resize(roi, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        input_data = np.expand_dims(rgb, 0).astype(np.uint8)

                        try:
                            in_stream.send(input_data)
                            out = landmark_stream.recv().flatten()
                            hailo_score = float(np.mean(np.abs(out)))
                            hailo_valid = hailo_score > HAILO_SCORE_TH
                        except:
                            hailo_valid = False

                    # MediaPipe
                    mediapipe_points = 0
                    rgb_big = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
                    results = hands_model.process(rgb_big)

                    if results.multi_hand_landmarks:
                        mediapipe_points = len(results.multi_hand_landmarks[0].landmark)
                        for hand_lm in results.multi_hand_landmarks:
                            h, w, _ = frame_vis.shape
                            for lm in hand_lm.landmark:
                                x = int(lm.x * w)
                                y = int(lm.y * h)
                                cv2.circle(frame_vis, (x, y), 3, (0, 255, 0), -1)

                    if not hailo_valid:
                        command = "No hand"

                    cv2.putText(frame_vis, f"Hailo score: {hailo_score:.4f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255 if hailo_valid else 0, 0 if hailo_valid else 255), 2)

                    # TCP
                    tcp_reconnected = 0
                    tcp_sent = 0

                    if command != last_sent_cmd:
                        proto_cmd = map_command_to_protocol(command)
                        tcp_reconnected = sender.send_command(proto_cmd)
                        tcp_sent = 1
                        last_sent_cmd = command

                    # Log
                    frame_time_ms = (time.time() - frame_start) * 1000
                    fps = 1000.0 / max(frame_time_ms, 0.001)

                    logger.log(
                        fps=fps,
                        frame_time_ms=frame_time_ms,
                        hailo_score=hailo_score,
                        hailo_valid=1 if hailo_valid else 0,
                        mediapipe_points=mediapipe_points,
                        command=command,
                        tcp_sent=tcp_sent,
                        tcp_reconnected=tcp_reconnected,
                        cx=cx if cx else -1,
                        cy=cy if cy else -1
                    )

                    # UI
                    cv2.imshow("Frame", frame_vis)
                    cv2.imshow("ROI", roi_vis)
                    cv2.imshow("Mask", mask)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        else:
            # Fara Hailo fallback
            while True:
                frame_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                command, frame_vis, roi, roi_vis, mask, (cx, cy) = process_frame_classic(frame)

                mediapipe_points = 0
                rgb_big = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
                results = hands_model.process(rgb_big)

                if results.multi_hand_landmarks:
                    mediapipe_points = 21
                    for hand_lm in results.multi_hand_landmarks:
                        h, w, _ = frame_vis.shape
                        for lm in hand_lm.landmark:
                            x = int(lm.x*w)
                            y = int(lm.y*h)
                            cv2.circle(frame_vis, (x, y), 3, (0,255,0), -1)

                # TCP
                tcp_reconnected = 0
                tcp_sent = 0
                if command != last_sent_cmd:
                    proto = map_command_to_protocol(command)
                    tcp_reconnected = sender.send_command(proto)
                    tcp_sent = 1
                    last_sent_cmd = command

                # Log
                frame_time_ms = (time.time() - frame_start)*1000
                fps = 1000/max(frame_time_ms,0.001)

                logger.log(
                    fps, frame_time_ms, 0.0, 0,
                    mediapipe_points, command, tcp_sent,
                    tcp_reconnected, cx if cx else -1, cy if cy else -1
                )

                cv2.imshow("Frame", frame_vis)
                cv2.imshow("ROI", roi_vis)
                cv2.imshow("Mask", mask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sender.close()
        logger.close()

        print("Resurse eliberate.")


if __name__ == "__main__":
    main()

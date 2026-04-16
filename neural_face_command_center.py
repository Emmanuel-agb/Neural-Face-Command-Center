import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------
# Utility helpers
# ---------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_point(p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[int, int]:
    return int(lerp(p1[0], p2[0], t)), int(lerp(p1[1], p2[1], t))


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def angle_deg(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    x, y = position
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_hud_box(frame: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], corner: int = 18) -> None:
    x2, y2 = x + w, y + h
    cv2.line(frame, (x, y), (x + corner, y), color, 2)
    cv2.line(frame, (x, y), (x, y + corner), color, 2)

    cv2.line(frame, (x2, y), (x2 - corner, y), color, 2)
    cv2.line(frame, (x2, y), (x2, y + corner), color, 2)

    cv2.line(frame, (x, y2), (x + corner, y2), color, 2)
    cv2.line(frame, (x, y2), (x, y2 - corner), color, 2)

    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 2)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    life: float
    color: Tuple[int, int, int]

    def update(self, dt: float) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        self.vx *= 0.985
        self.vy *= 0.985
        return self.life > 0

    def draw(self, frame: np.ndarray) -> None:
        if self.life <= 0:
            return
        alpha = clamp(self.life / 1.2, 0.15, 1.0)
        r = max(1, int(self.radius * alpha))
        color = tuple(int(c * alpha) for c in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), r, color, -1, cv2.LINE_AA)


class ParticleSystem:
    def __init__(self) -> None:
        self.particles: List[Particle] = []

    def burst(self, center: Tuple[int, int], count: int, color: Tuple[int, int, int], speed_min: float = 60, speed_max: float = 220) -> None:
        cx, cy = center
        for _ in range(count):
            ang = random.uniform(0, math.tau)
            speed = random.uniform(speed_min, speed_max)
            self.particles.append(
                Particle(
                    x=float(cx),
                    y=float(cy),
                    vx=math.cos(ang) * speed,
                    vy=math.sin(ang) * speed,
                    radius=random.randint(2, 5),
                    life=random.uniform(0.45, 1.2),
                    color=color,
                )
            )

    def ring(self, center: Tuple[int, int], radius: float, count: int, color: Tuple[int, int, int]) -> None:
        cx, cy = center
        for _ in range(count):
            ang = random.uniform(0, math.tau)
            px = cx + math.cos(ang) * radius
            py = cy + math.sin(ang) * radius
            self.particles.append(
                Particle(
                    x=px,
                    y=py,
                    vx=random.uniform(-35, 35),
                    vy=random.uniform(-35, 35),
                    radius=random.randint(1, 3),
                    life=random.uniform(0.3, 0.8),
                    color=color,
                )
            )

    def update(self, dt: float) -> None:
        self.particles = [p for p in self.particles if p.update(dt)]

    def draw(self, frame: np.ndarray) -> None:
        for p in self.particles:
            p.draw(frame)


# ---------------------------
# Gesture logic
# ---------------------------
def get_finger_states(hand_landmarks, handedness_label: str) -> List[bool]:
    lm = hand_landmarks.landmark

    thumb_tip = lm[4]
    thumb_ip = lm[3]

    if handedness_label == "Right":
        thumb_up = thumb_tip.x < thumb_ip.x
    else:
        thumb_up = thumb_tip.x > thumb_ip.x

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    return [thumb_up, index_up, middle_up, ring_up, pinky_up]


def classify_gesture(hand_landmarks, handedness_label: str) -> str:
    lm = hand_landmarks.landmark
    fingers = get_finger_states(hand_landmarks, handedness_label)
    total_up = sum(fingers)

    pinch_dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
    is_pinch = pinch_dist < 0.055

    if is_pinch:
        return "PINCH"
    if total_up >= 4:
        return "PALM"
    if total_up == 0:
        return "FIST"
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "PEACE"
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "POINT"
    return "NONE"


# ---------------------------
# Main app
# ---------------------------
class NeuralFaceCommandCenter:
    def __init__(self) -> None:
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.65,
        )
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.6,
        )

        self.particles = ParticleSystem()
        self.last_time = time.time()
        self.last_mode = "TRACK"
        self.mode_strength = 0.0
        self.mode_change_time = 0.0

        self.face_center: Optional[Tuple[int, int]] = None
        self.face_size: Tuple[int, int] = (220, 260)
        self.face_angle = 0.0

    def open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Close Zoom/Teams/Meet and try again.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def get_largest_face(self, detections, width: int, height: int):
        if not detections:
            return None

        best = None
        best_area = -1
        for detection in detections:
            box = detection.location_data.relative_bounding_box
            x = int(box.xmin * width)
            y = int(box.ymin * height)
            w = int(box.width * width)
            h = int(box.height * height)
            area = w * h
            if area > best_area:
                best_area = area
                best = detection
        return best

    def extract_face_info(self, detection, width: int, height: int):
        box = detection.location_data.relative_bounding_box
        x = int(clamp(box.xmin * width, 0, width - 1))
        y = int(clamp(box.ymin * height, 0, height - 1))
        w = int(clamp(box.width * width, 1, width))
        h = int(clamp(box.height * height, 1, height))
        x2 = int(clamp(x + w, 0, width - 1))
        y2 = int(clamp(y + h, 0, height - 1))
        w = x2 - x
        h = y2 - y

        keypoints = []
        for kp in detection.location_data.relative_keypoints:
            keypoints.append((int(kp.x * width), int(kp.y * height)))

        left_eye = keypoints[1]
        right_eye = keypoints[0]
        face_angle = angle_deg(left_eye, right_eye)
        center = (x + w // 2, y + h // 2)
        return x, y, w, h, center, keypoints, face_angle

    def smooth_face(self, center: Tuple[int, int], size: Tuple[int, int], angle: float) -> None:
        if self.face_center is None:
            self.face_center = center
            self.face_size = size
            self.face_angle = angle
            return

        self.face_center = lerp_point(self.face_center, center, 0.18)
        self.face_size = (
            int(lerp(self.face_size[0], size[0], 0.18)),
            int(lerp(self.face_size[1], size[1], 0.18)),
        )
        self.face_angle = lerp(self.face_angle, angle, 0.12)

    def extract_hands(self, hands_result, width: int, height: int):
        hands_data = []
        if not hands_result.multi_hand_landmarks or not hands_result.multi_handedness:
            return hands_data

        for hand_landmarks, handedness in zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness):
            label = handedness.classification[0].label
            gesture = classify_gesture(hand_landmarks, label)
            pts = [(int(l.x * width), int(l.y * height)) for l in hand_landmarks.landmark]
            palm = pts[9]
            hands_data.append(
                {
                    "label": label,
                    "gesture": gesture,
                    "points": pts,
                    "palm": palm,
                    "landmarks": hand_landmarks,
                }
            )
        return hands_data

    def decide_mode(self, hands_data, face_center: Optional[Tuple[int, int]]) -> str:
        gestures = [h["gesture"] for h in hands_data]

        if gestures.count("PALM") == 2:
            return "AURA"
        if "FIST" in gestures:
            return "LOCK"
        if "PEACE" in gestures:
            return "SCAN"
        if "PINCH" in gestures:
            return "PORTAL"
        if "POINT" in gestures and face_center is not None:
            return "FOCUS"
        return "TRACK"

    def render_tracking_hud(self, frame: np.ndarray, center: Tuple[int, int], size: Tuple[int, int], angle: float) -> None:
        cx, cy = center
        fw, fh = size
        x = cx - fw // 2 - 12
        y = cy - fh // 2 - 12
        draw_hud_box(frame, x, y, fw + 24, fh + 24, (90, 255, 180), 26)

        cv2.circle(frame, center, 4, (90, 255, 180), -1, cv2.LINE_AA)
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (90, 255, 180), 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (90, 255, 180), 1)

        # head tilt line
        line_r = max(fw, fh) // 2
        x1 = int(cx - math.cos(math.radians(angle)) * line_r * 0.35)
        y1 = int(cy - math.sin(math.radians(angle)) * line_r * 0.35)
        x2 = int(cx + math.cos(math.radians(angle)) * line_r * 0.35)
        y2 = int(cy + math.sin(math.radians(angle)) * line_r * 0.35)
        cv2.line(frame, (x1, y1), (x2, y2), (90, 255, 180), 1)

    def render_scan_mode(self, frame: np.ndarray, center: Tuple[int, int], size: Tuple[int, int], t: float) -> None:
        cx, cy = center
        fw, fh = size
        color = (255, 200, 70)

        self.render_tracking_hud(frame, center, size, self.face_angle)
        draw_hud_box(frame, cx - fw // 2 - 35, cy - fh // 2 - 35, fw + 70, fh + 70, color, 30)

        scan_progress = (math.sin(t * 3.1) + 1) * 0.5
        y_scan = int(cy - fh // 2 - 10 + scan_progress * (fh + 20))
        cv2.line(frame, (cx - fw // 2 - 25, y_scan), (cx + fw // 2 + 25, y_scan), color, 2)
        cv2.line(frame, (cx - fw // 2 - 10, y_scan + 3), (cx + fw // 2 + 10, y_scan + 3), (255, 240, 180), 1)

        draw_text(frame, "IDENTITY SCAN", (cx - fw // 2 - 10, cy - fh // 2 - 50), color, 0.72, 2)
        draw_text(frame, f"TILT {self.face_angle:+.1f} DEG", (cx + fw // 2 - 110, cy - fh // 2 - 18), color, 0.5, 1)
        draw_text(frame, "STATUS: TRACKING", (cx - fw // 2 - 8, cy + fh // 2 + 50), color, 0.55, 1)

        self.particles.ring(center, max(fw, fh) * 0.68, 6, color)

    def render_lock_mode(self, frame: np.ndarray, center: Tuple[int, int], size: Tuple[int, int], t: float) -> None:
        cx, cy = center
        fw, fh = size
        color = (70, 90, 255)
        pulse = 1.0 + 0.06 * math.sin(t * 7.0)
        rw = int(fw * 1.35 * pulse)
        rh = int(fh * 1.35 * pulse)

        draw_hud_box(frame, cx - rw // 2, cy - rh // 2, rw, rh, color, 28)
        cv2.circle(frame, center, int(max(fw, fh) * 0.62 * pulse), color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, int(max(fw, fh) * 0.35 * pulse), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 40, cy), (cx + 40, cy), color, 2)
        cv2.line(frame, (cx, cy - 40), (cx, cy + 40), color, 2)
        draw_text(frame, "LOCK-ON MODE", (cx - rw // 2 + 5, cy - rh // 2 - 10), color, 0.75, 2)
        draw_text(frame, "THREAT ANALYSIS ACTIVE", (cx - rw // 2 + 5, cy + rh // 2 + 25), color, 0.55, 1)

        for ang in range(0, 360, 45):
            rad = math.radians(ang + t * 70)
            x1 = int(cx + math.cos(rad) * max(fw, fh) * 0.52)
            y1 = int(cy + math.sin(rad) * max(fw, fh) * 0.52)
            x2 = int(cx + math.cos(rad) * max(fw, fh) * 0.82)
            y2 = int(cy + math.sin(rad) * max(fw, fh) * 0.82)
            cv2.line(frame, (x1, y1), (x2, y2), color, 1)

        self.particles.burst(center, 3, color, 40, 120)

    def render_aura_mode(self, frame: np.ndarray, center: Tuple[int, int], size: Tuple[int, int], t: float) -> None:
        cx, cy = center
        fw, fh = size
        color1 = (255, 240, 100)
        color2 = (120, 255, 255)

        for i in range(3):
            pulse = 1.0 + 0.08 * math.sin(t * 3.6 + i * 1.7)
            axes = (int(fw * (0.72 + i * 0.16) * pulse), int(fh * (0.82 + i * 0.14) * pulse))
            cv2.ellipse(frame, center, axes, self.face_angle, 0, 360, color1 if i % 2 == 0 else color2, 2, cv2.LINE_AA)

        for deg in range(0, 360, 30):
            a = math.radians(deg + t * 120)
            r = max(fw, fh) * 0.95
            px = int(cx + math.cos(a) * r)
            py = int(cy + math.sin(a) * r * 1.08)
            cv2.circle(frame, (px, py), 3, color2, -1, cv2.LINE_AA)

        draw_text(frame, "AURA SHIELD", (cx - fw // 2, cy - fh // 2 - 24), color2, 0.78, 2)
        draw_text(frame, "DOUBLE PALM DETECTED", (cx - fw // 2, cy + fh // 2 + 35), color1, 0.55, 1)

        self.particles.ring(center, max(fw, fh) * 0.92, 9, color1)

    def render_portal_mode(self, frame: np.ndarray, center: Tuple[int, int], size: Tuple[int, int], hands_data, t: float) -> None:
        cx, cy = center
        fw, fh = size
        color = (220, 120, 255)
        outer = int(max(fw, fh) * 0.95)
        inner = int(max(fw, fh) * 0.55)

        for i in range(4):
            radius = int(outer - i * 22 + 8 * math.sin(t * 5 + i))
            cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, inner, (80, 40, 150), 2, cv2.LINE_AA)
        draw_text(frame, "PORTAL CHANNEL", (cx - fw // 2, cy - fh // 2 - 24), color, 0.78, 2)

        pinch_hand = None
        for hand in hands_data:
            if hand["gesture"] == "PINCH":
                pinch_hand = hand
                break

        if pinch_hand:
            pinch_point = pinch_hand["points"][8]
            cv2.line(frame, pinch_point, center, color, 2)
            cv2.circle(frame, pinch_point, 10, color, 2, cv2.LINE_AA)
            self.particles.burst(pinch_point, 2, color, 20, 80)

        self.particles.ring(center, outer, 8, color)

    def render_focus_mode(self, frame: np.ndarray, center: Tuple[int, int], size: Tuple[int, int], hands_data, t: float) -> None:
        cx, cy = center
        fw, fh = size
        color = (100, 255, 120)
        self.render_tracking_hud(frame, center, size, self.face_angle)
        draw_text(frame, "FOCUS ASSIST", (cx - fw // 2, cy - fh // 2 - 24), color, 0.74, 2)

        for hand in hands_data:
            if hand["gesture"] == "POINT":
                tip = hand["points"][8]
                cv2.line(frame, tip, center, color, 2)
                cv2.circle(frame, tip, 10, color, 2, cv2.LINE_AA)
                cv2.circle(frame, center, int(max(fw, fh) * 0.78 + 12 * math.sin(t * 6)), color, 1, cv2.LINE_AA)
                break

        for i in range(6):
            a = math.radians(i * 60 + t * 50)
            x2 = int(cx + math.cos(a) * max(fw, fh) * 0.95)
            y2 = int(cy + math.sin(a) * max(fw, fh) * 0.95)
            cv2.line(frame, center, (x2, y2), color, 1)

    def render_face_keypoints(self, frame: np.ndarray, keypoints: List[Tuple[int, int]]) -> None:
        for pt in keypoints:
            cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (40, 40, 40), 1, cv2.LINE_AA)

    def render_hand_info(self, frame: np.ndarray, hands_data) -> None:
        for i, hand in enumerate(hands_data):
            self.mp_drawing.draw_landmarks(
                frame,
                hand["landmarks"],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 220, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )
            px, py = hand["palm"]
            draw_text(frame, f"{hand['label']} {hand['gesture']}", (px - 40, py - 20 - i * 4), (255, 255, 255), 0.5, 1)

    def overlay_header(self, frame: np.ndarray, fps: float, mode: str) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (w - 12, 78), (18, 18, 24), -1)
        frame[:] = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

        draw_text(frame, "NEURAL FACE COMMAND CENTER", (28, 42), (255, 230, 120), 0.9, 2)
        draw_text(frame, f"MODE: {mode}", (30, 68), (120, 255, 210), 0.6, 2)
        draw_text(frame, f"FPS: {fps:.1f}", (w - 135, 42), (255, 255, 255), 0.65, 2)
        draw_text(frame, "Gestures: PALM=Aura | FIST=Lock | PEACE=Scan | PINCH=Portal | POINT=Focus", (260, 68), (200, 200, 200), 0.47, 1)

    def render_mode_flash(self, frame: np.ndarray, dt_since_change: float) -> None:
        if dt_since_change > 0.24:
            return
        alpha = 1.0 - (dt_since_change / 0.24)
        flash = np.full_like(frame, 255)
        frame[:] = cv2.addWeighted(flash, 0.13 * alpha, frame, 1 - 0.13 * alpha, 0)

    def run(self) -> None:
        cap = self.open_camera()
        print("Controls:")
        print("  q = quit")
        print("  r = reset particles")
        print("  s = save screenshot")

        screenshot_id = 1

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                now = time.time()
                dt = max(1 / 120.0, now - self.last_time)
                self.last_time = now
                fps = 1.0 / dt

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                face_results = self.face_detector.process(rgb)
                hands_results = self.hands.process(rgb)
                rgb.flags.writeable = True

                detections = face_results.detections if face_results and face_results.detections else []
                face = self.get_largest_face(detections, width, height)
                hands_data = self.extract_hands(hands_results, width, height)
                mode = self.decide_mode(hands_data, self.face_center)

                if mode != self.last_mode:
                    self.last_mode = mode
                    self.mode_change_time = now
                    if self.face_center is not None:
                        burst_color = {
                            "TRACK": (120, 255, 210),
                            "AURA": (120, 255, 255),
                            "LOCK": (70, 90, 255),
                            "SCAN": (255, 200, 70),
                            "PORTAL": (220, 120, 255),
                            "FOCUS": (100, 255, 120),
                        }.get(mode, (255, 255, 255))
                        self.particles.burst(self.face_center, 18, burst_color, 50, 180)

                if face is not None:
                    x, y, w, h, center, keypoints, face_angle = self.extract_face_info(face, width, height)
                    self.smooth_face(center, (w, h), face_angle)

                    if self.face_center is not None:
                        self.render_tracking_hud(frame, self.face_center, self.face_size, self.face_angle)
                        self.render_face_keypoints(frame, keypoints)

                        t = time.time()
                        if mode == "SCAN":
                            self.render_scan_mode(frame, self.face_center, self.face_size, t)
                        elif mode == "LOCK":
                            self.render_lock_mode(frame, self.face_center, self.face_size, t)
                        elif mode == "AURA":
                            self.render_aura_mode(frame, self.face_center, self.face_size, t)
                        elif mode == "PORTAL":
                            self.render_portal_mode(frame, self.face_center, self.face_size, hands_data, t)
                        elif mode == "FOCUS":
                            self.render_focus_mode(frame, self.face_center, self.face_size, hands_data, t)
                        else:
                            draw_text(frame, "FACE TRACK ACTIVE", (self.face_center[0] - self.face_size[0] // 2, self.face_center[1] - self.face_size[1] // 2 - 24), (120, 255, 210), 0.72, 2)
                            self.particles.ring(self.face_center, max(self.face_size) * 0.72, 3, (120, 255, 210))
                else:
                    draw_text(frame, "No face detected - move into frame", (28, height - 28), (180, 180, 255), 0.7, 2)

                self.render_hand_info(frame, hands_data)
                self.particles.update(dt)
                self.particles.draw(frame)
                self.overlay_header(frame, fps, mode)
                self.render_mode_flash(frame, now - self.mode_change_time)

                cv2.imshow("Neural Face Command Center", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("r"):
                    self.particles = ParticleSystem()
                if key == ord("s"):
                    path = f"face_command_center_{screenshot_id}.png"
                    cv2.imwrite(path, frame)
                    print(f"Saved screenshot to {path}")
                    screenshot_id += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_detector.close()
            self.hands.close()


if __name__ == "__main__":
    try:
        app = NeuralFaceCommandCenter()
        app.run()
    except Exception as exc:
        print(f"Error: {exc}")

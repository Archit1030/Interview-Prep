"""
Posture Analysis — Full Multi-Engine Demo
Displays all active engines simultaneously:
  • MediaPipe Holistic  → face mesh + pose skeleton + hand landmarks
  • PostureAnalyzer     → shoulder angle, slouch, arms crossed, stability
  • StressAnalyzer      → blink rate, lip pursing, cognitive load
  • IntegrityChecker    → gaze pattern analysis
"""

import cv2
import mediapipe as mp
import time
from engine.vision_engine import VisionEngine

# ── MediaPipe drawing utils ───────────────────────────────────────────────────
mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

FACE_STYLE = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=1)
POSE_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=4)
CONN_STYLE = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)
HAND_STYLE = mp_drawing.DrawingSpec(color=(255, 80, 200), thickness=2, circle_radius=4)


# ── HUD drawing ───────────────────────────────────────────────────────────────
def label(frame, text, pos, color=(255, 255, 255), scale=0.55, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + bl + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame, metrics, fps):
    h, w = frame.shape[:2]
    posture = metrics.get("posture", {})
    stress  = metrics.get("stress",  {})
    integrity = metrics.get("integrity", {})

    # ── Title bar ────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.putText(frame, "POSTURE ANALYSIS  |  Full Multi-Engine Pipeline",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.1f}", (w - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # ── Left panel — Posture ─────────────────────────────────
    panel_x = 10
    y = 70
    label(frame, "[ POSTURE ENGINE ]", (panel_x, y), (0, 220, 255), 0.55, 2)

    y += 30
    angle = posture.get("shoulder_angle", 0.0)
    lean  = posture.get("is_leaning", False)
    label(frame, f"Shoulder Angle : {angle:+.1f} deg",
          (panel_x, y), (0, 0, 255) if lean else (0, 255, 0))

    y += 28
    slouch = posture.get("is_slouching", False)
    score  = posture.get("slouch_score", 0.0)
    label(frame, f"Slouch         : {'YES' if slouch else 'NO'}  score={score:.2f}",
          (panel_x, y), (0, 100, 255) if slouch else (0, 255, 0))

    y += 28
    crossed = posture.get("arms_crossed", False)
    label(frame, f"Arms Crossed   : {'YES !!!' if crossed else 'NO'}",
          (panel_x, y), (0, 0, 255) if crossed else (0, 255, 0))

    y += 28
    stab = posture.get("shoulder_stability", 1.0)
    rock = posture.get("rocking_score", 0.0)
    label(frame, f"Stability      : {stab:.2f}   Rocking: {rock:.2f}",
          (panel_x, y), (0, 255, 0) if stab > 0.7 else (0, 165, 255))

    # ── Left panel — Stress ──────────────────────────────────
    y += 44
    label(frame, "[ STRESS ENGINE ]", (panel_x, y), (0, 220, 255), 0.55, 2)

    y += 30
    blink_rate = stress.get("blink_rate", 0.0)
    cog        = stress.get("high_cognitive_load", False)
    label(frame, f"Blink Rate     : {blink_rate:.1f}/min {'[HIGH LOAD]' if cog else ''}",
          (panel_x, y), (0, 0, 255) if cog else (0, 255, 0))

    y += 28
    ear = stress.get("average_ear", 0.5)
    label(frame, f"Avg EAR        : {ear:.3f}",
          (panel_x, y), (200, 200, 200))

    y += 28
    lip     = stress.get("lip_pursing", False)
    lip_dur = stress.get("lip_purse_duration", 0.0)
    label(frame, f"Lip Pursing    : {'YES' if lip else 'NO'}  ({lip_dur:.1f}s)",
          (panel_x, y), (0, 100, 255) if lip else (0, 255, 0))

    y += 28
    lvl = stress.get("stress_level", "low")
    lvl_color = {"low": (0, 255, 0), "moderate": (0, 165, 255), "high": (0, 0, 255)}.get(lvl, (255, 255, 255))
    label(frame, f"Stress Level   : {lvl.upper()}", (panel_x, y), lvl_color)

    # ── Left panel — Integrity ───────────────────────────────
    y += 44
    label(frame, "[ INTEGRITY ENGINE ]", (panel_x, y), (0, 220, 255), 0.55, 2)

    y += 30
    flags   = integrity.get("cheat_flag_count", 0)
    warning = integrity.get("integrity_warning", False)
    iscore  = integrity.get("integrity_score", 1.0)
    label(frame, f"Integrity Score: {iscore:.2f}  Flags: {flags}",
          (panel_x, y), (0, 0, 255) if warning else (0, 255, 0))

    y += 28
    gx = integrity.get("gaze_x", 0.5)
    gy = integrity.get("gaze_y", 0.5)
    label(frame, f"Gaze Position  : ({gx:.2f}, {gy:.2f})",
          (panel_x, y), (200, 200, 200))

    if warning:
        y += 28
        label(frame, "⚠  INTEGRITY WARNING", (panel_x, y), (0, 0, 255), 0.6, 2)

    # ── Bottom badge ─────────────────────────────────────────
    good = not lean and not slouch and not crossed
    badge = "POSTURE: GOOD" if good else "POSTURE: NEEDS IMPROVEMENT"
    bcolor = (0, 200, 0) if good else (0, 0, 220)
    bx = w // 2 - 180
    cv2.rectangle(frame, (bx - 8, h - 48), (bx + 370, h - 8), (0, 0, 0), -1)
    cv2.putText(frame, badge, (bx, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, bcolor, 2, cv2.LINE_AA)

    # ── Controls ─────────────────────────────────────────────
    label(frame, "q=quit   r=reset baseline", (10, h - 15), (140, 140, 140), 0.45)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  POSTURE ANALYSIS  —  Full Multi-Engine Pipeline")
    print("=" * 60)

    # VisionEngine handles all analysis
    engine = VisionEngine()

    # Separate MediaPipe Holistic instance just for drawing landmarks
    holistic_draw = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        refine_face_landmarks=True
    )

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nSit upright for ~2 seconds to calibrate slouch baseline.")
    print("Controls:  q = quit   r = reset baseline\n")

    frame_count = 0
    t0 = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read frame")
            break

        frame = cv2.flip(frame, 1)

        # ── 1. Draw landmarks via dedicated holistic instance ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        draw_results = holistic_draw.process(rgb)
        rgb.flags.writeable = True

        # Face mesh
        if draw_results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, draw_results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_styles.get_default_face_mesh_contours_style()
            )

        # Pose skeleton
        if draw_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, draw_results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=CONN_STYLE
            )

        # Hands
        if draw_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, draw_results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                HAND_STYLE, CONN_STYLE
            )
        if draw_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, draw_results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                HAND_STYLE, CONN_STYLE
            )

        # ── 2. Run VisionEngine for all metrics ────────────────
        metrics = engine.analyze_frame(frame)

        # ── 3. Draw HUD ────────────────────────────────────────
        frame_count += 1
        if frame_count % 15 == 0:
            fps = frame_count / (time.time() - t0)

        draw_hud(frame, metrics, fps)

        cv2.imshow("Posture Analysis — Full Multi-Engine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            engine.posture_analyzer.reset()
            engine.signal_smoother.reset()
            frame_count = 0
            t0 = time.time()
            print("🔄 Baseline reset — sit upright to recalibrate")

    cap.release()
    cv2.destroyAllWindows()
    holistic_draw.close()
    engine.release()

    elapsed = time.time() - t0
    print(f"\nDone. {frame_count} frames in {elapsed:.1f}s  ({frame_count/elapsed:.1f} FPS avg)")


if __name__ == "__main__":
    main()

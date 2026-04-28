"""
Live Demo: Posture Analysis with Webcam
Shows HolisticProcessor + SignalSmoother + PostureAnalyzer working together in real-time
"""

import cv2
import numpy as np
import time
from engine.holistic_processor import HolisticProcessor
from engine.signal_smoother import SignalSmoother
from engine.analyzers.posture_analyzer import PostureAnalyzer


def draw_text_with_background(frame, text, position, font_scale=0.6, thickness=2, 
                              text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # Draw background rectangle
    cv2.rectangle(frame, 
                 (x - 5, y - text_height - 5), 
                 (x + text_width + 5, y + baseline + 5), 
                 bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def draw_posture_overlay(frame, metrics, fps):
    """Draw posture metrics overlay on the frame."""
    h, w = frame.shape[:2]
    
    # Title
    draw_text_with_background(frame, "LIVE POSTURE ANALYSIS", (10, 30), 
                             font_scale=0.8, thickness=2, 
                             text_color=(0, 255, 255), bg_color=(0, 0, 0))
    
    # FPS
    draw_text_with_background(frame, f"FPS: {fps:.1f}", (w - 120, 30), 
                             font_scale=0.6, text_color=(0, 255, 0))
    
    # Shoulder Angle
    y_offset = 70
    angle_color = (0, 0, 255) if metrics.is_leaning else (0, 255, 0)
    draw_text_with_background(frame, f"Shoulder Angle: {metrics.shoulder_angle:.1f}deg", 
                             (10, y_offset), text_color=angle_color)
    
    if metrics.is_leaning:
        draw_text_with_background(frame, "WARNING: Leaning!", 
                                 (10, y_offset + 30), text_color=(0, 0, 255))
        y_offset += 30
    
    # Slouch Detection
    y_offset += 40
    slouch_color = (0, 0, 255) if metrics.is_slouching else (0, 255, 0)
    draw_text_with_background(frame, f"Slouch Score: {metrics.slouch_score:.2f}", 
                             (10, y_offset), text_color=slouch_color)
    
    if metrics.is_slouching:
        draw_text_with_background(frame, "WARNING: Slouching!", 
                                 (10, y_offset + 30), text_color=(0, 0, 255))
        y_offset += 30
    
    # Arms Crossed
    y_offset += 40
    arms_color = (255, 165, 0) if metrics.arms_crossed else (0, 255, 0)
    arms_text = "Arms: CROSSED" if metrics.arms_crossed else "Arms: Open"
    draw_text_with_background(frame, arms_text, (10, y_offset), text_color=arms_color)
    
    # Stability
    y_offset += 40
    stability_color = (0, 255, 0) if metrics.shoulder_stability > 0.7 else (255, 165, 0)
    draw_text_with_background(frame, f"Stability: {metrics.shoulder_stability:.2f}", 
                             (10, y_offset), text_color=stability_color)
    
    draw_text_with_background(frame, f"Rocking: {metrics.rocking_score:.2f}", 
                             (10, y_offset + 30), text_color=stability_color)
    
    # Status indicators (bottom right)
    status_y = h - 100
    draw_text_with_background(frame, "STATUS:", (w - 200, status_y), 
                             font_scale=0.5, text_color=(200, 200, 200))
    
    status_y += 25
    posture_status = "GOOD" if not metrics.is_leaning and not metrics.is_slouching else "NEEDS IMPROVEMENT"
    status_color = (0, 255, 0) if posture_status == "GOOD" else (0, 165, 255)
    draw_text_with_background(frame, f"Posture: {posture_status}", 
                             (w - 200, status_y), font_scale=0.5, text_color=status_color)
    
    status_y += 25
    stability_status = "STABLE" if metrics.shoulder_stability > 0.7 else "FIDGETING"
    stability_color = (0, 255, 0) if stability_status == "STABLE" else (255, 165, 0)
    draw_text_with_background(frame, f"Body: {stability_status}", 
                             (w - 200, status_y), font_scale=0.5, text_color=stability_color)


def draw_landmarks(frame, pose_landmarks):
    """Draw key pose landmarks on the frame."""
    if not pose_landmarks or len(pose_landmarks) < 25:
        return
    
    h, w = frame.shape[:2]
    
    # Draw shoulders
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    
    ls_x, ls_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
    rs_x, rs_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
    
    # Draw shoulder line
    cv2.line(frame, (ls_x, ls_y), (rs_x, rs_y), (0, 255, 0), 3)
    cv2.circle(frame, (ls_x, ls_y), 8, (0, 255, 0), -1)
    cv2.circle(frame, (rs_x, rs_y), 8, (0, 255, 0), -1)
    
    # Draw nose
    nose = pose_landmarks[0]
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    cv2.circle(frame, (nose_x, nose_y), 8, (255, 0, 0), -1)
    
    # Draw line from nose to shoulder midpoint
    mid_x = (ls_x + rs_x) // 2
    mid_y = (ls_y + rs_y) // 2
    cv2.line(frame, (nose_x, nose_y), (mid_x, mid_y), (255, 255, 0), 2)
    
    # Draw wrists
    left_wrist = pose_landmarks[15]
    right_wrist = pose_landmarks[16]
    
    lw_x, lw_y = int(left_wrist.x * w), int(left_wrist.y * h)
    rw_x, rw_y = int(right_wrist.x * w), int(right_wrist.y * h)
    
    cv2.circle(frame, (lw_x, lw_y), 8, (255, 0, 255), -1)
    cv2.circle(frame, (rw_x, rw_y), 8, (255, 0, 255), -1)


def main():
    """Main demo function."""
    print("=" * 70)
    print("LIVE POSTURE ANALYSIS DEMO")
    print("=" * 70)
    print("\nInitializing components...")
    
    # Initialize all components
    holistic_processor = HolisticProcessor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_frame_skip=True,
        target_fps=15.0
    )
    
    signal_smoother = SignalSmoother(
        freq=30.0,
        min_cutoff=1.0,
        beta=0.0,
        d_cutoff=1.0
    )
    
    posture_analyzer = PostureAnalyzer(
        shoulder_angle_threshold=15.0,
        slouch_threshold=0.05,  # More sensitive slouch detection
        rock_threshold=0.02
    )
    
    print("\n✅ All components initialized!")
    print("\n" + "=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print("• Sit in front of your webcam")
    print("• Try different postures:")
    print("  - Sit up straight (good posture)")
    print("  - Lean to one side (shoulder angle)")
    print("  - Slouch forward (slouch detection)")
    print("  - Cross your arms (arms crossed detection)")
    print("  - Rock side to side (stability detection)")
    print("\n• Press 'q' to quit")
    print("• Press 'r' to reset analyzer")
    print("=" * 70)
    
    input("\nPress Enter to start...")
    
    # Open webcam (CAP_MSMF works on Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n🎥 Webcam opened! Starting analysis...\n")
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_start = time.time()
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # 1. Process with HolisticProcessor
            holistic_results = holistic_processor.process_frame(frame)
            
            # 2. Smooth landmarks with SignalSmoother
            smoothed_pose, smoothed_face, smoothed_left_hand, smoothed_right_hand = signal_smoother.smooth_landmarks(
                pose_landmarks=holistic_results.pose_landmarks,
                face_landmarks=holistic_results.face_landmarks,
                left_hand_landmarks=holistic_results.left_hand_landmarks,
                right_hand_landmarks=holistic_results.right_hand_landmarks,
                timestamp=holistic_results.timestamp
            )
            
            # 3. Analyze posture with PostureAnalyzer
            posture_metrics = posture_analyzer.analyze(
                pose_landmarks=smoothed_pose,
                timestamp=holistic_results.timestamp
            )
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
            
            # Draw visualizations
            draw_landmarks(frame, smoothed_pose)
            draw_posture_overlay(frame, posture_metrics, fps)
            
            # Add instructions at bottom
            h, w = frame.shape[:2]
            draw_text_with_background(frame, "Press 'q' to quit | Press 'r' to reset", 
                                     (10, h - 20), font_scale=0.5, 
                                     text_color=(200, 200, 200), bg_color=(0, 0, 0))
            
            # Show frame
            cv2.imshow('Live Posture Analysis - Interview Mirror', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('r'):
                print("\n🔄 Resetting analyzer...")
                posture_analyzer.reset()
                signal_smoother.reset()
                frame_count = 0
                start_time = time.time()
                print("✅ Reset complete!")
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                arms_status = "CROSSED" if posture_metrics.arms_crossed else "Open"
                print(f"Frame {frame_count}: "
                      f"Angle={posture_metrics.shoulder_angle:.1f}° | "
                      f"Slouch={posture_metrics.slouch_score:.2f} | "
                      f"Arms={arms_status} | "
                      f"Stability={posture_metrics.shoulder_stability:.2f} | "
                      f"FPS={fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        
        # Get component stats
        holistic_stats = holistic_processor.get_performance_stats()
        smoother_stats = signal_smoother.get_stats()
        
        print(f"\nHolistic Processor:")
        print(f"  - FPS: {holistic_stats['fps']}")
        print(f"  - Avg process time: {holistic_stats['avg_process_time_ms']:.2f}ms")
        
        print(f"\nSignal Smoother:")
        print(f"  - Active filters: {smoother_stats['total_filters']}")
        
        print("\n✅ Demo completed successfully!")
        print("=" * 70)
        
        cap.release()
        cv2.destroyAllWindows()
        holistic_processor.release()


if __name__ == "__main__":
    main()

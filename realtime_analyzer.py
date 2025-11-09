# ==============================================================================
# realtime_analyzer.py (The "Eyes" and Main Application)
#
# This script handles all video processing and GUI display. It imports the
# ThreatDetector from your unmodified threat_algorithm.py to perform analysis.
# ==============================================================================

import cv2
import time
from ultralytics import YOLO
import supervision as sv

# Import your "Brain" from our other file
from threat_algorithm import ThreatDetector

def main():
    MODEL_PATH = 'visDrone_pretrained.pt'
    VIDEO_SOURCE_PATH = 'test_parking_lot.mp4'
    VIDEO_OUTPUT_PATH = 'output_final_analysis.mp4'
    CONFIDENCE_THRESHOLD = 0.6
    SAVE_OUTPUT_VIDEO = True

    print("--- Entity Real-Time Analyzer GUI Execution Start ---")
    
    cap_in = cv2.VideoCapture(VIDEO_SOURCE_PATH)
    if not cap_in.isOpened():
        print(f"Error: Could not open video file: {VIDEO_SOURCE_PATH}")
        return
        
    width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    
    cap_out = None
    if SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap_out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))
    
    # Load the "Eyes"
    model = YOLO(MODEL_PATH)
    class_names = model.model.names
    print("Visual model loaded.")
    
    # Initialize your "Brain" EXACTLY as it is designed
    threat_detector = ThreatDetector()
    print("Analytical Core initialized.")
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    tracker = sv.ByteTrack()

    cv2.namedWindow("Entity - Real-Time Analysis", cv2.WINDOW_NORMAL)

    frame_number = 0
    while cap_in.isOpened():
        ret, frame = cap_in.read()
        if not ret:
            break
        frame_number += 1
        current_time = frame_number / fps

        # Perform detection with performance optimization
        results = model(frame, device='mps', verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = tracker.update_with_detections(detections)

        # Bridge: Format data and send it to the "Brain"
        detections_for_threat_model = [
            {'obj_id': tid, 'label': class_names[cid], 'bbox': xyxy}
            for xyxy, cid, tid
            in zip(tracked_detections.xyxy, tracked_detections.class_id, tracked_detections.tracker_id)
        ]
        frame_data_for_threat_model = {'timestamp': current_time, 'detections': detections_for_threat_model}
        
        # Get alerts back from the "Brain"
        alerts = threat_detector.process_frame_data(frame_data_for_threat_model)
        
        annotated_frame = frame.copy()
        labels = [f"#{tid} {class_names[cid]}" for cid, tid in zip(tracked_detections.class_id, tracked_detections.tracker_id)]
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

        if alerts:
            for alert in alerts:
                if alert['obj_id'] in tracked_detections.tracker_id:
                    idx = list(tracked_detections.tracker_id).index(alert['obj_id'])
                    xyxy = tracked_detections.xyxy[idx]
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    alert_text = f"ALERT: {alert['threat_type']} ({alert['probability']:.0%})"
                    cv2.putText(annotated_frame, alert_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 3)

        cv2.imshow("Entity - Real-Time Analysis", annotated_frame)
        if SAVE_OUTPUT_VIDEO:
            cap_out.write(annotated_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' key pressed. Exiting analysis.")
            break

    cap_in.release()
    if SAVE_OUTPUT_VIDEO:
        cap_out.release()
    cv2.destroyAllWindows()
    
    print("\n--- Processing Complete ---")
    if SAVE_OUTPUT_VIDEO:
        print(f"Output video with threat analysis saved to: {VIDEO_OUTPUT_PATH}")

if __name__ == "__main__":
    main()

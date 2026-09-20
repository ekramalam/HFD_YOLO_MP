#Human-Fall-Detection-Yolov12-MediaPipe\fall_detection_app.py
#!/usr/bin/env python3
import os
import sys
import argparse


from pathlib import Path
from turtle import color
from fall_detector_engine import FallEngine


import fall_detector_engine

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fall Detection System')
    
   
    parser.add_argument('--model', type=str, default='yolov12n1.pt',
                      help='Path to YOLOv12 model file (default: yolov12n1.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='Detection confidence threshold (0-1)')
    
    # Video source options
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam, or path to video file)')
    
    # Detection parameters
    parser.add_argument('--fall-threshold', type=float, default=0.1,
                      help='Threshold for fall detection sensitivity (0-1)')
    parser.add_argument('--angle-threshold', type=float, default=45,
                      help='Threshold for body angle in degrees (0-90)')
    
    # Output options
    parser.add_argument('--save-falls', action='store_true',
                      help='Save frames when falls are detected')

    #added this argument i.e --dataset-dir
    parser.add_argument('--dataset-dir', type=str, default=None,
                    help='Root directory containing videos in subfolders')

    parser.add_argument('--output-dir', type=str, default='fall_snapshots',
                      help='Directory to save fall snapshots (if --save-falls is used)')
    
    return parser.parse_args()




def run_cli_mode(args):
    """Run the system in command line mode."""
    import cv2
    import time
    
    print("Starting Fall Detection System in CLI mode")
    print(f"Using model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    
    # Initialize fall detector
    fall_detector_engine = FallEngine(
        model_path=args.model,
        confidence=args.conf
    )
    
    # Set detection parameters if provided
    if args.fall_threshold:
        fall_detector_engine.fall_threshold = args.fall_threshold
    if args.angle_threshold:
        fall_detector_engine.angle_threshold = args.angle_threshold
    
    # Open video source
    if args.source.isdigit():
        source_id = int(args.source)
        print(f"Opening webcam with ID: {source_id}")
        cap = cv2.VideoCapture(source_id, cv2.CAP_DSHOW)  # Try DSHOW backend on Windows
    else:
        print(f"Opening video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Print camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video source opened with resolution: {width}x{height}, FPS: {fps}")
    
    # Initialize FPS calculation
    frame_count = 0
    start_time = time.time()
    last_fps_update = start_time
    
    # Add tracking for fallen people and total fall count
    active_falls = set()  # Track currently fallen people by ID
    total_falls = 0       # Count of total unique falls
    
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to receive frame. Exiting...")
                break
            
            frame_count += 1
            
            # Process the frame
            output_frame, fall_detected, fall_data = fall_detector_engine.process_frame(frame)
            
            # Update fall tracking
            if fall_detected and 'person_ids' in fall_data and 'fallen_ids' in fall_data:
                # Get people who are currently fallen
                currently_fallen = set(fall_data['fallen_ids'])
                
                # Find newly fallen people (not already in active_falls)
                new_falls = currently_fallen - active_falls
                if new_falls:
                    # Increment total falls count by number of new falls
                    total_falls += len(new_falls)
                
                # Update our tracking of active falls
                active_falls = currently_fallen
                
                # Remove IDs of people who are no longer in the frame
                if 'person_ids' in fall_data:
                    all_people = set(fall_data['person_ids'])
                    active_falls &= all_people  # Keep only IDs still present
            
            # Display statistics and status
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Update FPS every second
            if current_time - last_fps_update >= 1.0:
                current_fps = frame_count / elapsed_time
                last_fps_update = current_time
                
                # Display basic stats
                print(f"\rFPS: {current_fps:.1f} | People: {fall_data['person_count']} | Current falls: {len(active_falls)} | Total falls: {total_falls}", end='')
                
                # If fall detected, print more details
                if fall_detected:
                    # fall_type = fall_data["fall_type"] or "unknown type"
                    print(f"\nFALL DETECTED: {fall_type}")
            
            # Display the output frame
            cv2.imshow('Fall Detection', output_frame)
            
            # Handle keyboard commands
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        print("\nReleasing resources...")
        cap.release()
        cv2.destroyAllWindows()


#added this function i.e run_dataset_mode
def run_dataset_mode(args):
    from pathlib import Path
    import cv2


    ds_root_dir = Path(args.dataset_dir) 
    # input dataset directory containing subfolders with videos 

    dataset_name = ds_root_dir.name

      # fall_detector_engine = FallEngine(args.model, args.conf)

    fall_detector_engine = FallEngine(
        args.model,
        args.conf,
        args.fall_threshold
    )

    fall_detector_engine.angle_threshold = args.angle_threshold

    from pathlib import Path
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parent directory (fixed)
    parent_output_dir = Path("outputs")
    parent_output_dir.mkdir(parents=True, exist_ok=True)


    # Build experiment-specific output folder name
    experiment_folder_name = (
        f"{args.output_dir}_"
        f"conf{args.conf:.2f}_"
        f"ft{args.fall_threshold:.2f}_"
        f"at{int(args.angle_threshold)}_"
        # f"ar{fall_detector_engine.aspect_ratio:.2f}_"
        f"AnChTh{fall_detector_engine.angle_change_threshold:.0f}_"
        f"cf{fall_detector_engine.confirm_frames}_"
        f"vdF{fall_detector_engine.vertical_drop_frames}_"
        f"aChF{fall_detector_engine.angle_change_frames}_"
        f"{Path(args.model).stem}"
        f"_{timestamp}"
    )

    
    result_dir = parent_output_dir / experiment_folder_name
    # result_dir = Path(experiment_folder_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    

  


  


  

    experiment_stats = {
    "total_videos": 0,
    "adl_videos": 0,
    "fall_videos": 0,
    "tp": 0,
    "tn": 0,
    "fp": 0,
    "fn": 0,
    }


    for video_path in ds_root_dir.rglob("*.mp4"):
        print(f"\nProcessing: {video_path}")
        video_id = video_path.stem   # "01.mp4" → "01"
        fall_detector_engine.current_video_name = video_path.name

        rel_path = video_path.relative_to(ds_root_dir)

        video_path_str = str(video_path).lower()

        is_gt_fall = "fall" in video_path_str
        is_gt_adl  = "adl"  in video_path_str

        ground_truth_label = "fall" if is_gt_fall else "adl"


        experiment_stats["total_videos"] += 1
        if is_gt_fall:
            experiment_stats["fall_videos"] += 1
        else:
            experiment_stats["adl_videos"] += 1

        # if hasattr(fall_detector_engine, "baseline_height"):
        #     del fall_detector_engine.baseline_height

        fall_detector_engine.baseline_height = None
        # Reset detector state for each video
        fall_detector_engine.prev_poses.clear()
        fall_detector_engine.fallen_person_ids.clear()
        fall_detector_engine.fall_detected = False
        fall_detector_engine.video_fall_detected = False
        # fall_detector_engine.latency_frozen = False
        fall_detector_engine.fall_start_time = None
        fall_detector_engine.consecutive_fall_frames = 0
        fall_detector_engine.last_fall_person_id = None

        video_fall_detected = False

        fall_start_logged = False
        fall_start_time_s = None




        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping video (cannot open): {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30
        fall_detector_engine.fps = fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rel_path = video_path.relative_to(ds_root_dir)
        save_dir = result_dir / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        output_video_path = save_dir / f"{video_id}_annotated.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (width, height)
        )


        frame_no = 0
        frame_id = "F0"

        


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            frame_id = f"F{frame_no}"
            # video_time_ms = (frame_no / fps) * 1000
            # fps = fps if fps > 0 else 30
            video_time_s = frame_no / fps


            out_frame, fall_detected, fall_data = fall_detector_engine.process_frame(frame)

          


        


            if fall_detected:
                video_fall_detected = True
                if not fall_start_logged:
                    fall_start_time_s = video_time_s
                    fall_start_logged = True

                # Get vertical drop safely
                vertical_drop = 0
                if len(fall_detector_engine.prev_poses) > 5:
                    curr = fall_detector_engine.prev_poses[-1]
                    prev = fall_detector_engine.prev_poses[-6]
                    # vertical_drop = curr["mid_shoulder_y"] - prev["mid_shoulder_y"]

                criteria_text = " | ".join(fall_detector_engine.last_fall_criteria)
                criteria_text += f" | VerticalDrop={vertical_drop:.3f}"



        

            video_writer.write(out_frame)
        video_writer.release()
        cap.release()
        if frame_no == 0:
            print(f"Warning: No frames processed for video {video_path}")

        import gc

        del frame
        del out_frame
        gc.collect()


        if is_gt_fall and video_fall_detected:
            experiment_stats["tp"] += 1
        elif is_gt_adl and not video_fall_detected:
            experiment_stats["tn"] += 1
        elif is_gt_adl and video_fall_detected:
            experiment_stats["fp"] += 1
        elif is_gt_fall and not video_fall_detected:
            experiment_stats["fn"] += 1

    tp = experiment_stats["tp"]
    tn = experiment_stats["tn"]
    fp = experiment_stats["fp"]
    fn = experiment_stats["fn"]

    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1_score  = 2 * precision * recall / max(precision + recall, 1e-6)

    import json
    from datetime import datetime

   

    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "ds_root_dir": str(ds_root_dir),
            "model": args.model,
        },

        "detection_parameters": {
            "yolo_confidence": fall_detector_engine.confidence,
            "head_position_override_threshold": fall_detector_engine.top_threshold,
             "aro_threshold": fall_detector_engine.aro_threshold,
            "head_ground_threshold": fall_detector_engine.head_ground_threshold,

            "motion_parameters": {
                "position_smoothing_alpha": fall_detector_engine.alpha_position,
                 "velocity_window_frames": fall_detector_engine.velocity_window_frames
            },


            "mediapipe_pose": {
                "static_image_mode": fall_detector_engine.static_image_mode,
                "model_complexity": fall_detector_engine.model_complexity,
                "enable_segmentation": fall_detector_engine.enable_segmentation,
                "min_detection_confidence": fall_detector_engine.min_detection_confidence,
                "min_tracking_confidence": fall_detector_engine.min_tracking_confidence
            },



            "angle_threshold": fall_detector_engine.angle_threshold,
            # "aspect_ratio_threshold": fall_detector_engine.aspect_ratio,
            "fall_threshold_vertical_drop": fall_detector_engine.fall_threshold,
            "angle_change_threshold": fall_detector_engine.angle_change_threshold,

            "confirm_frames": fall_detector_engine.confirm_frames,
            "vertical_drop_frames": fall_detector_engine.vertical_drop_frames,
            "angle_change_frames": fall_detector_engine.angle_change_frames,

            "prev_pose_buffer_size": fall_detector_engine.prev_pose_buffer_size
        },

        "dataset_statistics": experiment_stats,

        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        },

        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }
    }


    # with open(result_dir / "summary.json", "w") as f:
    #     json.dump(summary, f, indent=4)


    json_filename = f"{dataset_name}_{timestamp}.json"
    with open(result_dir / json_filename, "w") as f:
        json.dump(summary, f, indent=4)






def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Check if the YOLOv12 model file exists
    if not os.path.exists(args.model) and not args.model.startswith('yolov12'):
        print(f"Warning: Model file '{args.model}' not found.")
        print("The system will attempt to download it if it's a standard YOLOv12 model.")
    
  
    # Run in the selected mode
    if args.dataset_dir:
        run_dataset_mode(args)
    # elif args.mode == 'dashboard':
    #     run_dashboard_mode(args)
    else:
        run_cli_mode(args)


if __name__ == "__main__":
    main() 

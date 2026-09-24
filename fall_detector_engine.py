
import cv2
import numpy as np
import os
import datetime
import json
import math


def create_logger(log_file_path='logs/fall_detection_log.json'):

  
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Initialize the log file if it doesn't exist
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            json.dump([], f)
    
    def log_event(event_type, event_data):
   
        # Read existing logs
        try:
            with open(log_file_path, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
        
        # Add timestamp to event data
        event_data['timestamp'] = datetime.datetime.now().isoformat()
        event_data['event_type'] = event_type
        
        # Append new log
        logs.append(event_data)
        
        # Write updated logs back to file
        with open(log_file_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    return log_event


def draw_skeleton(frame, landmarks, connections, box=None):

    h, w = frame.shape[:2]
    
    if box:
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        
        
        pixel_landmarks = []
        for lm in landmarks:
            x, y = int(lm[0] * box_w), int(lm[1] * box_h)
            pixel_landmarks.append((x + x1, y + y1))
    else:
   
        pixel_landmarks = []
        for lm in landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            pixel_landmarks.append((x, y))

    for connection in connections:
        start_idx, end_idx = connection
        if (start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks)):
            cv2.line(
                frame,
                pixel_landmarks[start_idx],
                pixel_landmarks[end_idx],
                (245, 66, 230),
                2
            )
    
  
    for point in pixel_landmarks:
        cv2.circle(frame, point, 3, (245, 117, 66), -1)
    
    return frame


def calculate_angle(point1, point2, point3):

  
    a = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    b = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    

    dot_product = np.dot(a, b)
    
    
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    
    
    cos_angle = dot_product / (mag_a * mag_b)
    
    
    cos_angle = max(-1, min(1, cos_angle))
    
    angle = math.degrees(math.acos(cos_angle))
    return angle


def get_euclidean_distance(point1, point2):
    
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def is_valid_pose(landmarks):
    
   
    if len(landmarks) < 33:  # MediaPipe has 33 pose landmarks
        return False
    
   
    key_indices = [0, 11, 12, 23, 24, 27, 28]  
    for idx in key_indices:
        if idx < len(landmarks) and landmarks[idx][3] < 0.5:  
            return False
    
    return True


def detect_overlapping_people(person_boxes, iou_threshold=0.3):

    caution_indices = []
    
    for i in range(len(person_boxes)):
        for j in range(i+1, len(person_boxes)):
            box1 = person_boxes[i]
            box2 = person_boxes[j]
            
            
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
               
                continue
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            iou = intersection / union
            
            if iou > iou_threshold:
                if i not in caution_indices:
                    caution_indices.append(i)
                if j not in caution_indices:
                    caution_indices.append(j)
    
    return caution_indices 

def draw_text_panel(frame, lines, position="top_left"):
    

    overlay = frame.copy()
    h, w = frame.shape[:2]

    padding = 10
    line_height = 20
    panel_width = 500
    panel_height = padding * 2 + line_height * len(lines)

    # Choose position
    if position == "top_left":
        x, y = 10, 10
    elif position == "top_right":
        x, y = w - panel_width - 10, 10
    elif position == "bottom_left":
        x, y = 10, h - panel_height - 10
    elif position == "bottom_right":
        x, y = w - panel_width - 10, h - panel_height - 10
    else:
        x, y = 10, 10

   
    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_width, y + panel_height),
        (40, 40, 40),   
        -1
    )

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)



    for i, item in enumerate(lines):

        if isinstance(item, tuple):
            text, color = item
        else:
            text = item
            color = (255, 255, 255)

        cv2.putText(
            frame,
            text,
            (x + padding, y + padding + (i + 1) * line_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return frame



from utils import draw_skeleton
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math
import time
import torch
import os
from utils import draw_text_panel

class FallEngine:
    


    def __init__(self, model_path='yolov12n1.pt', confidence=0.5, fall_threshold=0.1):
        
        self.confidence = confidence
        self.fall_threshold = fall_threshold
        self.bbox_padding_ratio = 0.15   
        self.draw_debug_padding_box = True
        self.last_fall_person_id = None
        self.fc_display_snapshot = None
        self.current_video_name = ""
        
        



        self.static_image_mode = False
        self.model_complexity = 2
        self.enable_segmentation = False
        self.min_detection_confidence = 0.55
        self.min_tracking_confidence = 0.5
        self.baseline_height = None

        self.aro_threshold = 1.25
        self.knee_ankle_angle_th = 15
        self.knee_ankle_angle_th2 = 22
        self.knee_hip_angle_th = 18
        self.knee_hip_dist_th = 0.1
        self.ankle_knee_dist_th = 0.2
        self.alpha_position = 0.9
        self.velocity_window_frames = 5
        self.velocity_threshold = 400
        self.prev_pose_buffer_size = 20

        self.mediapipe_padding_ratio = 0.00   
        
    
        self.last_fall_criteria = []   
        self.all_fall_criteria = []   
        self.consecutive_fall_frames = 0
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                self.device = "cuda"
                
             
                torch.cuda.init()
                x = torch.tensor([1.0], device="cuda")
                print(f"Test tensor device: {x.device}")
            else:
                self.device = "cpu"
                print("Using CPU for detection")
            
          
            print(f"Loading YOLO12 model: {model_path}")
            
           
            try:
                from ultralytics.utils.downloads import attempt_download
                downloaded_path = attempt_download(model_path)
                if downloaded_path:
                    print(f"Downloaded model to: {downloaded_path}")
                    model_path = str(downloaded_path) 
                else:
                    print("Download returned Nothing, using original model path.")
            except Exception as download_error:
                print(f"Model download error (continuing with original model path): {download_error}")
            
        
            print(f"Using model path: {model_path}")
            self.model = YOLO(model_path, task='detect')
            self.model.to(self.device)
            print(f"Successfully loaded YOLO12 model on {self.device}")
            
      
            model_device = next(self.model.parameters()).device
            print(f"Model confirmed on the device: {model_device}")
            
      
            self.person_class_id = 0
            self.confidence = confidence
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        

        
        self.angle_threshold = 45  
        self.angle_change_threshold = 15  
      
        self.confirm_frames = 2
     
        self.top_threshold = 0.4 

       
        self.head_ground_threshold = 0.7 
        self.vertical_drop_frames = 8
        self.angle_change_frames = 5
        self.fps = 30   

        self.prev_poses = [] 
        

        self.motion_history = []  
        
    
        self.next_person_id = 1
        self.person_trackers = {}  
        self.fallen_person_ids = set()  
        self.video_fall_detected = False




    def landmarks_inside_box(self, landmarks, person_box):

        x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom = person_box[:4]

        bbox_width = x2_bbox_right - x1_bbox_left
        bbox_height = y2_bbox_bottom - y1_bbox_top

        pad_x = bbox_width * self.bbox_padding_ratio
        pad_y = bbox_height * self.bbox_padding_ratio

        padded_x1 = x1_bbox_left - pad_x
        padded_y1 = y1_bbox_top - pad_y
        padded_x2 = x2_bbox_right + pad_x
        padded_y2 = y2_bbox_bottom + pad_y

      
        check_ids = [0, 11, 12, 23, 24, 25, 26, 27, 28]

        for idx in check_ids:

            lx = x1_bbox_left + landmarks[idx][0] * bbox_width
            ly = y1_bbox_top + landmarks[idx][1] * bbox_height

            if lx < padded_x1 or lx > padded_x2 or ly < padded_y1 or ly > padded_y2:
                return False

        return True
        

        
    def detect_person(self, frame):
     
  
        
        try:
         
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != 'uint8':
                    frame = cv2.convertScaleAbs(frame)
                
         
            results = self.model(frame, 
                                verbose=False, 
                                conf=self.confidence,
                                device=self.device)
            
            person_boxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
           

                    if cls == self.person_class_id and conf > self.confidence:
                        x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom = box.xyxy[0].cpu().numpy().astype(int)
                        person_boxes.append((x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom, conf))
            
            return person_boxes
        except Exception as e:
            print(f"Error in detecting_person: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def analyze_pose(self, frame, person_box):

     

        x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom = person_box[:4]

        frame_h, frame_w = frame.shape[:2]

     
        pad_x = int((x2_bbox_right - x1_bbox_left) * self.mediapipe_padding_ratio)
        pad_y = int((y2_bbox_bottom - y1_bbox_top) * self.mediapipe_padding_ratio)

        x1_padded_left = max(0, x1_bbox_left - pad_x)
        y1_padded_top = max(0, y1_bbox_top - pad_y)
        x2_padded_right = min(frame_w, x2_bbox_right + pad_x)
        y2_padded_bottom = min(frame_h, y2_bbox_bottom + pad_y)

  
        person_img = frame[y1_padded_top:y2_padded_bottom, x1_padded_left:x2_padded_right]


        
        if person_img.size == 0:
            return None, None
        
   
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
     
        results = self.pose.process(image=rgb_img)
        
        if not results.pose_landmarks:
            return None, None
        

        

        landmarks = []

        crop_w = x2_padded_right - x1_padded_left
        crop_h = y2_padded_bottom - y1_padded_top

        for landmark in results.pose_landmarks.landmark:

          
            lx_frame = x1_padded_left + landmark.x * crop_w
            ly_frame = y1_padded_top + landmark.y * crop_h

            lx = (lx_frame - x1_bbox_left) / (x2_bbox_right - x1_bbox_left)
            ly = (ly_frame - y1_bbox_top) / (y2_bbox_bottom - y1_bbox_top)

            landmarks.append((lx, ly, landmark.z, landmark.visibility))

    
        pose_features = self.calculate_pose_features(landmarks, person_box)
        
        return landmarks, pose_features
    
    
    def calculate_pose_features(self, landmarks, person_box):

        x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom = person_box[:4]
        bbox_width = x2_bbox_right - x1_bbox_left
        bbox_height = y2_bbox_bottom - y1_bbox_top

     
        head_y_px = y1_bbox_top + landmarks[0][1] * bbox_height
        ankle_y_px = y1_bbox_top + ((landmarks[27][1] + landmarks[28][1]) / 2) * bbox_height

        body_height_ank_to_head = abs(ankle_y_px - head_y_px)


        box_width = x2_bbox_right - x1_bbox_left
        box_height = y2_bbox_bottom - y1_bbox_top

       

        if box_width > 0:
            aspect_ratio = box_height / box_width
        else:
            aspect_ratio = 0


        frame_h = self.frame_height

        dt = 1 / self.fps

        

        if not landmarks:
            return None
        
    
        key_points = [0, 11, 12, 23, 24, 25, 26, 27, 28]



        nose_y_box = landmarks[0][1]         
        nose_frame_y = y1_bbox_top + nose_y_box * bbox_height
        nose_y_norm = nose_frame_y / frame_h
        
  
        key_landmarks = [landmarks[i][:2] for i in key_points if i < len(landmarks)]
        
        if len(key_landmarks) < len(key_points):
            return None
        
      
        head_y = landmarks[0][1]
        left_ankle_y = landmarks[27][1]
        right_ankle_y = landmarks[28][1]
        ankle_y = (left_ankle_y + right_ankle_y) / 2
        height = abs(ankle_y - head_y)
        
     

        mid_shoulder_x = x1_bbox_left + ((landmarks[11][0] + landmarks[12][0]) / 2) * bbox_width
        mid_shoulder_y = y1_bbox_top + ((landmarks[11][1] + landmarks[12][1]) / 2) * bbox_height

        alpha = self.alpha_position

        if self.prev_poses:
            prev_y = self.prev_poses[-1]["mid_shoulder_y"]
            mid_shoulder_y = alpha * mid_shoulder_y + (1 - alpha) * prev_y

        mid_hip_x_norm = (landmarks[23][0] + landmarks[24][0]) / 2 
        mid_hip_y_norm = (landmarks[23][1] + landmarks[24][1]) / 2

        mid_hip_x = x1_bbox_left + ((landmarks[23][0] + landmarks[24][0]) / 2) * bbox_width
        mid_hip_y = y1_bbox_top + ((landmarks[23][1] + landmarks[24][1]) / 2) * bbox_height

  
        sh_y = mid_shoulder_y
        hp_y = mid_hip_y

    
        left_sh = landmarks[11][0]
        right_sh = landmarks[12][0]

        sh_left_x = x1_bbox_left + left_sh * bbox_width
        sh_right_x = x1_bbox_left + right_sh * bbox_width

        sh_width = abs(sh_right_x - sh_left_x)

        sh_height = abs(hp_y - sh_y)

        if sh_width > 1e-6:
            aspect_ratio_sh = sh_height / sh_width
        else:
            aspect_ratio_sh = 0


        feet_x_norm = (landmarks[27][0] + landmarks[28][0]) / 2
        feet_y_norm = (landmarks[27][1] + landmarks[28][1]) / 2

        feet_pos = (feet_x_norm, feet_y_norm)

        mid_hipX_mid_shldrX = mid_hip_x - mid_shoulder_x
        mid_hipY_mid_shldrY = mid_hip_y - mid_shoulder_y

        angle_signed = math.degrees(math.atan2(mid_hipX_mid_shldrX, mid_hipY_mid_shldrY))
        angle = abs(angle_signed)
        
      
        shoulder_pos = (mid_shoulder_x, mid_shoulder_y)
        hip_pos = (mid_hip_x, mid_hip_y)
      
        left_most_x = min(landmarks[11][0], landmarks[23][0], landmarks[25][0], landmarks[27][0])
        right_most_x = max(landmarks[12][0], landmarks[24][0], landmarks[26][0], landmarks[28][0])
        top_most_y = min(landmarks[0][1], landmarks[11][1], landmarks[12][1])
        bottom_most_y = max(landmarks[27][1], landmarks[28][1])
        
        bbox_width = right_most_x - left_most_x
        bbox_height = bottom_most_y - top_most_y
       
        
      

        
       
       
       

        hip_to_feet_distance = math.sqrt(
        (mid_hip_x_norm - feet_x_norm)**2 +
        (mid_hip_y_norm - feet_y_norm)**2
        )
        


        standing_arOverride = aspect_ratio > self.aro_threshold

       
        left_knee_x, left_knee_y = landmarks[25][:2]
        left_ankle_x, left_ankle_y = landmarks[27][:2]

        dx_l = left_ankle_x - left_knee_x
        dy_l = left_ankle_y - left_knee_y

        angle_knee_ankle_left = abs(math.degrees(math.atan2(dx_l, dy_l)))


       
        right_knee_x, right_knee_y = landmarks[26][:2]
        right_ankle_x, right_ankle_y = landmarks[28][:2]

        dx_r = right_ankle_x - right_knee_x
        dy_r = right_ankle_y - right_knee_y

        angle_knee_ankle_right = abs(math.degrees(math.atan2(dx_r, dy_r)))

        angle_knee_ankle = min(angle_knee_ankle_left, angle_knee_ankle_right)

   
        left_hip_x, left_hip_y = landmarks[23][:2]

      

        dx_lkn_hp = left_knee_x - left_hip_x
        dy_lk_hp = left_knee_y - left_hip_y

        angle_knee_hip_left = abs(math.degrees(math.atan2(dx_lkn_hp, dy_lk_hp)))


       
        right_hip_x, right_hip_y = landmarks[24][:2]

        dx_rh = right_knee_x - right_hip_x
        dy_rh = right_knee_y -right_hip_y

        angle_knee_hip_right = abs(math.degrees(math.atan2(dx_rh, dy_rh)))

        angle_knee_hip = min(angle_knee_hip_left, angle_knee_hip_right)

        vertical_knee_hip_dist_left = left_knee_y-left_hip_y
        vertical_knee_hip_dist_right = right_knee_y- right_hip_y

        vertical_knee_hip_dist = min(vertical_knee_hip_dist_left, vertical_knee_hip_dist_right)

    
        is_standing = (
            (angle_knee_ankle < self.knee_ankle_angle_th and angle_knee_hip < self.knee_hip_angle_th)
           
        )

   

       
        left_knee_y = landmarks[25][1]
        left_ankle_y = landmarks[27][1]

        right_knee_y = landmarks[26][1]
        right_ankle_y = landmarks[28][1]

        left_knee_ankle_dist = abs(left_knee_y - left_ankle_y)
        right_knee_ankle_dist = abs(right_knee_y - right_ankle_y)

        left_ankle_knee_vdist = left_ankle_y - left_knee_y
        right_ankle_knee_vdist = right_ankle_y - right_knee_y

        ankle_knee_vdist = min(left_ankle_knee_vdist, right_ankle_knee_vdist)



        knee_ankle_distance = (
            min(left_knee_ankle_dist,right_knee_ankle_dist)
                
        )


        
       
        velocity_y = 0  
        velocity_x = 0  
      
        vertical_drop_norm = 0
        vertical_drop_px = 0
        
        if self.prev_poses:
            prev = self.prev_poses[-1]
       

            frames = self.velocity_window_frames

            if len(self.prev_poses) >= frames:
                prev_pose = self.prev_poses[-frames]
                velocity_y = (mid_shoulder_y - prev_pose["mid_shoulder_y"]) / (frames * dt)
            else:
                velocity_y = 0
            
        
            velocity_x = (mid_shoulder_x - prev["mid_shoulder_x"]) / dt
            
     
        raw_shoulder_to_hip = math.sqrt(
            (mid_shoulder_x - mid_hip_x)**2 +
            (mid_shoulder_y - mid_hip_y)**2
        )


        shoulder_to_hip_distance = raw_shoulder_to_hip
        spike_detected = False

          

        if self.prev_poses:
            prev_height = self.prev_poses[-1]["shoulder_to_hip_distance"]
            if abs(raw_shoulder_to_hip - prev_height) > prev_height * 0.2:
                spike_detected = True
                shoulder_to_hip_distance = prev_height

          

            vel = abs(velocity_y)
            if vel < 10:
                vel = 0

       
            prev_height = max(prev_height, 1e-6)
          
        shoulder_vis = min(landmarks[11][3], landmarks[12][3])
        hip_vis = min(landmarks[23][3], landmarks[24][3])
        valid_visibility = (shoulder_vis >= 0.5 and hip_vis >= 0.5)

     
        current_h = shoulder_to_hip_distance


        if self.baseline_height is None:
            self.baseline_height = current_h

        else:
            if current_h <= self.baseline_height * 1.15 and valid_visibility:
       
                self.baseline_height = max(self.baseline_height, current_h)
        
        
        
      
        height_ratio = shoulder_to_hip_distance / self.baseline_height
        height_ratio = min(height_ratio, 1)

          

 
            
       
     


        

        if len(self.prev_poses) >= self.vertical_drop_frames:

            prev_pose = self.prev_poses[-self.vertical_drop_frames]

            vertical_drop_px = mid_shoulder_y - prev_pose["mid_shoulder_y"]

            if  self.frame_height > 0:
                vertical_drop_norm = vertical_drop_px / self.frame_height


       

          
       
     
        
        features = {
            "height": height,
            "body_height_ank_to_head": body_height_ank_to_head,
            "angle": angle,
            "angle_signed": angle_signed,
            "angle_knee_ankle_left": angle_knee_ankle_left,
            "angle_knee_ankle_right": angle_knee_ankle_right,
            "angle_knee_ankle": angle_knee_ankle,
            "angle_knee_hip_left": angle_knee_hip_left,
            "angle_knee_hip_right": angle_knee_hip_right,
            "angle_knee_hip": angle_knee_hip,
            "ankle_knee_vdist_left": left_ankle_knee_vdist,
            "ankle_knee_vdist_right": right_ankle_knee_vdist,
            "ankle_knee_vdist": ankle_knee_vdist,
            "vertical_knee_hip_dist_left": vertical_knee_hip_dist_left,
            "vertical_knee_hip_dist_right": vertical_knee_hip_dist_right,
            "vertical_knee_hip_dist": vertical_knee_hip_dist,
            "mid_hipX_mid_shldrX": mid_hipX_mid_shldrX,
            "nose_y": nose_y_norm,
            "box_width": box_width,
            "box_height": box_height,
            "aspect_ratio": aspect_ratio,
            "aspect_ratio_sh": aspect_ratio_sh,
            "standing_ar": standing_arOverride,
            "is_standing": is_standing,
            "shoulder_y": mid_shoulder_y,
            "velocity_y": velocity_y,
            "velocity_x": velocity_x,
            "mid_shoulder_y": mid_shoulder_y,
            "mid_shoulder_x": mid_shoulder_x,
            "mid_hip_y": mid_hip_y,
            "mid_hip_x": mid_hip_x,
            "shoulder_pos": shoulder_pos,
            "hip_pos": hip_pos,
            "feet_pos": feet_pos,
            "vertical_drop_norm": vertical_drop_norm,
            "vertical_drop_px": vertical_drop_px,
            "height_ratio": height_ratio,
            "knee_ankle_distance": knee_ankle_distance,
            "left_knee_ankle_dist": left_knee_ankle_dist,
            "right_knee_ankle_dist": right_knee_ankle_dist,
            "shoulder_to_hip_distance": shoulder_to_hip_distance,
            "shoulder_to_hip_raw": raw_shoulder_to_hip,
            "shoulder_spike": spike_detected,
            "hip_to_feet_distance": hip_to_feet_distance,
            "timestamp": time.time()
        }
        
     
        
        return features
    
   
            
  
    def detect_fall(self, pose_features, person_id=None):
  
       

        if not pose_features:
            return False

  
        self.last_fall_criteria = []
        self.all_fall_criteria = []

        angle = (pose_features["angle"])
    
        height_ratio = pose_features.get("height_ratio", 1)
        vertical_drop_norm = pose_features.get("vertical_drop_norm", 0)
        nose_y = pose_features.get("nose_y", 1)
        knee_ankle = pose_features.get("knee_ankle_distance", 1)
       
        angle_change = 0
        vel = pose_features.get("velocity_y", 0)

     


        if nose_y < self.top_threshold:
            self.last_fall_criteria.append("OVERRIDE: Head near top of frame")
            self.consecutive_fall_frames = 0
            self.last_fall_person_id = None
            return False

       

        
        standing_arOverride = pose_features.get("standing_ar", False)
        
        aspect_ratio = pose_features.get("aspect_ratio", 0)

        


        if standing_arOverride:
            self.last_fall_criteria.append(
                f"OVERRIDE: Standing AR Override ({aspect_ratio:.2f})"
            )
            self.consecutive_fall_frames = 0
            self.last_fall_person_id = None
            return False



        is_standing = pose_features.get("is_standing", False)


  



        if is_standing:
            self.last_fall_criteria.append("OVERRIDE: Standing posture")
            self.consecutive_fall_frames = 0
            self.last_fall_person_id = None
            return False    


        knee_ankle_angle = pose_features.get("angle_knee_ankle", 0)
        ankle_knee_vdist = pose_features.get("ankle_knee_vdist", 0)
        knee_hip_vdist = pose_features.get("vertical_knee_hip_dist", 0)



        if (
            knee_ankle_angle < self.knee_ankle_angle_th2 and
            ankle_knee_vdist > self.ankle_knee_dist_th and
            knee_hip_vdist > self.knee_hip_dist_th
        ):
            self.last_fall_criteria.append(
                f"OVERRIDE: straight standing legs "
                f"(KAang={knee_ankle_angle:.1f}, AKvd={ankle_knee_vdist:.2f}, KHvd={knee_hip_vdist:.2f})"
            )
            self.consecutive_fall_frames = 0
            self.last_fall_person_id = None
            return False

      


        

        fall_criteria_met = 0

      
        c1_met = angle > self.angle_threshold
        if c1_met:
            fall_criteria_met += 1
            self.last_fall_criteria.append(f"C1: Angle > threshold ({angle:.1f}°)")
        self.all_fall_criteria.append({
            "name": "C1",
            "description": "Angle > threshold",
            "threshold": self.angle_threshold,
            "value": angle,
            "met": c1_met
        })

    

       

        c3_met = vertical_drop_norm > self.fall_threshold

        if c3_met:
            fall_criteria_met += 1
            self.last_fall_criteria.append(f"C3: Vertical drop > threshold ({vertical_drop_norm:.3f})")
        self.all_fall_criteria.append({
            "name": "C3",
            "description": "Vertical drop > threshold",
            "threshold": self.fall_threshold,
            "value": vertical_drop_norm,
            "met": c3_met
        })
        



          
        c4_met = False
   

        angle_change = 0
        if len(self.prev_poses) >= self.angle_change_frames:
            prev = self.prev_poses[-self.angle_change_frames]
       


            mid_hipX_mid_shldrX = pose_features["mid_hipX_mid_shldrX"]

            currentAngSigned = pose_features["angle_signed"]
            PrvsAngSigned = prev["angle_signed"]

            if mid_hipX_mid_shldrX > 0:  
                angle_change = currentAngSigned - PrvsAngSigned
            else:      
                angle_change = PrvsAngSigned - currentAngSigned

           
            c4_met = angle_change > self.angle_change_threshold
            if c4_met:
                fall_criteria_met += 1
                self.last_fall_criteria.append(f"C4: Angle change >  {self.angle_change_threshold} ({angle_change:.1f}°)")
        self.all_fall_criteria.append({
            "name": "C4",
            "description": "Angle change > {self.angle_change_threshold}",
            "threshold": self.angle_change_threshold,
            "value": angle_change,
            "met": c4_met
        })

        c5_met = height_ratio < 0.5

        if c5_met:
            fall_criteria_met += 1
            self.last_fall_criteria.append(
                f"C5: Height ratio < 0.5 ({height_ratio:.2f})"
            )

        self.all_fall_criteria.append({
            "name": "C5",
            "description": "Height ratio < 0.5",
            "threshold": 0.5,
            "value": height_ratio,
            "met": c5_met
        })

       
        c6_met = knee_ankle < 0.06

 
        if c6_met:
            fall_criteria_met += 1
            self.last_fall_criteria.append(
                f"C6: Knee-Ankle distance < 0.06 ({knee_ankle:.3f})"
            )

        self.all_fall_criteria.append({
            "name": "C6",
            "description": "Knee-Ankle distance < 0.06",
            "threshold": 0.06,
            "value": knee_ankle,
            "met": c6_met
        })


        



       
        c7_met = nose_y >= self.head_ground_threshold

        if c7_met:
            fall_criteria_met += 1
            self.last_fall_criteria.append(
                f"C7: HeadY >= {self.head_ground_threshold} ({nose_y:.2f})"
            )

        self.all_fall_criteria.append({
            "name": "C7",
            "description": "Head near ground",
            "threshold": self.head_ground_threshold,
            "value": nose_y,
            "met": c7_met
        })


      
        c_vel_met = vel > self.velocity_threshold

        if c_vel_met:
            fall_criteria_met += 1
            self.last_fall_criteria.append(f"Crit_vel: Velocity > {self.velocity_threshold} ({vel:.1f})")

        self.all_fall_criteria.append({
            "name": "Crit_vel",
            "description": f"Velocity > {self.velocity_threshold} px/s",
            "threshold": self.velocity_threshold,
            "value": vel,
            "met": c_vel_met
        })

     

        self.last_fall_criteria.append(f"TOTAL_CRITERIA={fall_criteria_met}")



        


    


        if fall_criteria_met >= 3:
            if self.last_fall_person_id == person_id:
                self.consecutive_fall_frames += 1
            else:
                self.consecutive_fall_frames = 1
                self.last_fall_person_id = person_id
        else:
            self.consecutive_fall_frames = 0
            self.last_fall_person_id = None

        if self.consecutive_fall_frames >= self.confirm_frames:
            pose_features["fall_criteria_met"] = fall_criteria_met
            return True
        
        

        return False

       

    def process_frame(self, frame):
     

     
        self.frame_height, self.frame_width = frame.shape[:2]
     
        output_frame = frame.copy()
        
      
        person_boxes = self.detect_person(frame)
        
    
        box_to_id = self.track_persons(frame, person_boxes)
        
      
        current_person_ids = list(box_to_id.values())
        
        falls_detected = False
        fall_data = {
            "person_count": len(person_boxes),
            "fall_detected": False,
            "person_boxes": person_boxes,
            "critical_points": [],
            "person_ids": current_person_ids,
            "fallen_ids": []
        }
        
       
        pose_data = []
        
       
        for i, box in enumerate(person_boxes):
            person_id = box_to_id.get(i)
         
            x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom, yolo_conf = box
            
          
            landmarks, pose_features = self.analyze_pose(frame, box)


            if not landmarks or not pose_features:
                continue

            if not self.landmarks_inside_box(landmarks, box):
                continue
       
            pose_data.append((person_id, landmarks, pose_features, (x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom)))

            self.prev_poses.append(pose_features)
            if len(self.prev_poses) > self.prev_pose_buffer_size:
                self.prev_poses.pop(0)

     
        for person_id, landmarks, pose_features, (x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom) in pose_data:
         
            cv2.rectangle(
                output_frame,
                (x1_bbox_left, y1_bbox_top),
                (x2_bbox_right, y2_bbox_bottom),
                (0, 255, 0),
                2
            )


         
            pad_x = int((x2_bbox_right - x1_bbox_left) * self.mediapipe_padding_ratio)
            pad_y = int((y2_bbox_bottom - y1_bbox_top) * self.mediapipe_padding_ratio)

            x1_padded_left = max(0, x1_bbox_left - pad_x)
            y1_padded_top = max(0, y1_bbox_top - pad_y)
            x2_padded_right = min(self.frame_width, x2_bbox_right + pad_x)
            y2_padded_bottom = min(self.frame_height, y2_bbox_bottom + pad_y)

            cv2.rectangle(output_frame,(x1_padded_left,y1_padded_top),(x2_padded_right,y2_padded_bottom),(255,150,0),1)

        
        
            if self.draw_debug_padding_box:

                pad_x = int((x2_bbox_right - x1_bbox_left) * self.bbox_padding_ratio)
                pad_y = int((y2_bbox_bottom - y1_bbox_top) * self.bbox_padding_ratio)

                cv2.rectangle(
                    output_frame,
                    (x1_bbox_left - pad_x, y1_bbox_top - pad_y),
                    (x2_bbox_right + pad_x, y2_bbox_bottom + pad_y),
                    (200,200,200),
                    1
                )


          
            if landmarks:
                draw_skeleton(
                output_frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                box=(x1_bbox_left, y1_bbox_top, x2_bbox_right, y2_bbox_bottom)
                )

         
            if person_id in self.fallen_person_ids:
                continue
                
      
            if not landmarks or not pose_features:
                continue
                
        
            angle = pose_features.get("angle", 0)

        
            is_fall = self.detect_fall(pose_features, person_id)
         
            velocity_y = pose_features.get("velocity_y", 0)
          

            vertical_drop_norm = pose_features.get("vertical_drop_norm", 0)

           
            if is_fall:
           
                self.video_fall_detected = True

                fall_data["fall_detected"] = True
          
                falls_detected = True
                self.fc_display_snapshot = pose_features.get("fall_criteria_met", 0)

                if person_id not in fall_data["fallen_ids"]:
                    fall_data["fallen_ids"].append(person_id)
                    self.fallen_person_ids.add(person_id)

          

        



     
        COLOR_PASS = (0,255,0)       
        COLOR_OVERRIDE = (0,255,255)
        COLOR_FAIL = (250,250,250) 
        COLOR_INFO = (200,200,200)   

        debug_lines = []

        dt = 1 / self.fps
        frames = self.velocity_window_frames

        debug_lines.append(f"File: {self.current_video_name}")
        debug_lines.append(f"Person Count: {fall_data['person_count']}")
    
        

        for person_id, _, pose_features, _ in pose_data:

            if not pose_features:
                continue

            angle = pose_features.get("angle", 0)
           
            vert_drop = pose_features.get("vertical_drop_norm", 0)
            vel = pose_features.get("velocity_y", 0)
         
            height_ratio = pose_features.get("height_ratio", 1)

            debug_lines.append(f"Person ID: {person_id}")

      
            shoulder_y = pose_features.get("shoulder_y",0)

       

            mid_hipY_mid_shldrY = vel * frames * dt
            prev_y = shoulder_y - mid_hipY_mid_shldrY

      

           

          

            nose = pose_features.get("nose_y", 1)

            head_override = nose < self.top_threshold

            if head_override:
                color = (0,255,255)   
            else:
                color = (250,250,250)

            debug_lines.append(
                (f"HPO: {nose:.2f} < {self.top_threshold} [{'YES' if head_override else 'NO'}]", color)
            )


            aspect = pose_features.get("aspect_ratio", 0)
            w = pose_features.get("box_width", 0)
            h = pose_features.get("box_height", 0)

     


            aspect = pose_features.get("aspect_ratio", 0)
            standing_ar = pose_features.get("standing_ar", False)
            is_standing = pose_features.get("is_standing", False)

            color_ar = COLOR_OVERRIDE if standing_ar else COLOR_FAIL
            color_st = COLOR_OVERRIDE if is_standing else COLOR_FAIL

            debug_lines.append(
                (f"ARO: {aspect:.2f} >  {self.aro_threshold} [{'YES' if standing_ar else 'NO'}] | H: {h:.0f} | W: {w:.0f} ", color_ar)
            )

            
            kaAn_l = pose_features.get("angle_knee_ankle_left",0)
            kaAn_r = pose_features.get("angle_knee_ankle_right",0)
            kaAn = pose_features.get("angle_knee_ankle",0)

            khAng_l = pose_features.get("angle_knee_hip_left",0)
            khAng_r = pose_features.get("angle_knee_hip_right",0)
            khAng = pose_features.get("angle_knee_hip",0)

          

            debug_lines.append(
                (f"AngKA L:{kaAn_l:.1f} R:{kaAn_r:.1f} Min:{kaAn:.1f}", COLOR_INFO)
            )

            debug_lines.append(
                (f"AngKH L:{khAng_l:.1f} R:{khAng_r:.1f} Min:{khAng:.1f}", COLOR_INFO)
            )


            debug_lines.append(
            (
            f"VLO: {kaAn:.1f}<{self.knee_ankle_angle_th} {khAng:.1f}<{self.knee_hip_angle_th} "
            f"[{'YES' if is_standing else 'NO'}]",
            color_st
            )
            )

            

            vkhDist_l = pose_features.get("vertical_knee_hip_dist_left", 0)
            vkhDist_r = pose_features.get("vertical_knee_hip_dist_right", 0)
            knee_hip_vdist = pose_features.get("vertical_knee_hip_dist", 0)
            vakDist_l = pose_features.get("ankle_knee_vdist_left", 0)
            vakDist_r = pose_features.get("ankle_knee_vdist_right", 0)
            ankle_knee_vdist = pose_features.get("ankle_knee_vdist", 0)


            debug_lines.append(
                (f"D_KA_O L:{vakDist_l:.2f} R:{vakDist_r:.2f} Min:{ankle_knee_vdist:.2f} > {self.ankle_knee_dist_th}",
                COLOR_INFO)
            )

            debug_lines.append(
                (f"D_KH L:{vkhDist_l:.2f} R:{vkhDist_r:.2f} Min:{knee_hip_vdist:.2f} > {self.knee_hip_dist_th}",
                COLOR_INFO)
            )

            knee_ankle_angle = pose_features.get("angle_knee_ankle",0)
           

            override_straight_leg = (
                knee_ankle_angle < self.knee_ankle_angle_th2 and
                ankle_knee_vdist > self.ankle_knee_dist_th and
                knee_hip_vdist > self.knee_hip_dist_th
            )


            override_active = (
                head_override or
                standing_ar or
                is_standing or
                override_straight_leg
            )

            color = COLOR_OVERRIDE if override_straight_leg else COLOR_FAIL

            debug_lines.append(
            (
            f"LAO: AngKA:{knee_ankle_angle:.1f}<{self.knee_ankle_angle_th2} "
            f"D_KA_O:{ankle_knee_vdist:.2f}> {self.ankle_knee_dist_th} "
            f"D_KH:{knee_hip_vdist:.2f}> {self.knee_hip_dist_th} "
            f"[{'YES' if override_straight_leg else 'NO'}]",
            color
            )
            )

            




   

            if override_active:

                debug_lines.append(
                    ("Fall Criteria Skipped (Override Active)", (0,165,255))
                )

              

                continue

          
            c1 = angle > self.angle_threshold
            color = (0,255,0) if c1 else (250,250,250)
            debug_lines.append((f"C1 Ang: {angle:.1f} > {self.angle_threshold} [{'YES' if c1 else 'NO'}]", color))



            
            vert_drop = pose_features.get("vertical_drop_norm", 0)
            raw_drop = pose_features.get("vertical_drop_px", 0)
          
            body_height_ank_to_head = pose_features.get("body_height_ank_to_head", 1)
           

            c3 = vert_drop > self.fall_threshold
            color = COLOR_PASS if c3 else COLOR_FAIL

        

            debug_lines.append(
            (
            f"C2 VerDrp: {vert_drop:.3f} ({raw_drop:.1f}/{self.frame_height:.1f}) > {self.fall_threshold}",
            color
            )
            )


          


   

            height_ratio = pose_features.get("height_ratio", 1)
            shoulder_hip = pose_features.get("shoulder_to_hip_distance", 0)

            shoulder_hip_raw = pose_features.get("shoulder_to_hip_raw", 0)
            spike = pose_features.get("shoulder_spike", False)

            baseline = self.baseline_height if self.baseline_height else 0.0001

            c5 = height_ratio < 0.5
            color = COLOR_PASS if c5 else COLOR_FAIL

            aspect_sh = pose_features.get("aspect_ratio_sh", 0)

    
            debug_lines.append(
                (f"shoulder_to_hip_dist_Spike: {'YES' if spike else 'NO'}", COLOR_INFO)
            )

            debug_lines.append(
            (
            f"C3 HR: {height_ratio:.2f} ({shoulder_hip:.1f}/{baseline:.1f}) < 0.5 | H_raw: {shoulder_hip_raw:.1f}"
            f"[{'YES' if c5 else 'NO'}]",
            color
            )
            )

        
            angle_change = 0
            mid_hipX_mid_shldrX = 0
            currentAngSigned = 0
            PrvsAngSigned = 0
           
            if len(self.prev_poses) >= self.angle_change_frames:
                prev = self.prev_poses[-self.angle_change_frames]
               

                mid_hipX_mid_shldrX = pose_features["mid_hipX_mid_shldrX"]

                currentAngSigned = pose_features["angle_signed"]
                PrvsAngSigned = prev["angle_signed"]

                if mid_hipX_mid_shldrX > 0:
                    angle_change = currentAngSigned - PrvsAngSigned
                else:
                    angle_change = PrvsAngSigned - currentAngSigned

            c4 = angle_change > self.angle_change_threshold
            color = COLOR_PASS if c4 else COLOR_FAIL

            debug_lines.append(
            (
            f"C4 AngCh: {angle_change:.1f} ({currentAngSigned:.1f}-{PrvsAngSigned:.1f}) "
            f"> {self.angle_change_threshold} "
            f"[{'YES' if c4 else 'NO'}] M_hipX-M_shdX:{mid_hipX_mid_shldrX:.0f}",
            color
            )
            )

         
            knee_ank_dis = pose_features.get("knee_ankle_distance", 1)
            left_k = pose_features.get("left_knee_ankle_dist", 0)
            right_k = pose_features.get("right_knee_ankle_dist", 0)

            c6 = knee_ank_dis < 0.06
            color = COLOR_PASS if c6 else COLOR_FAIL

            debug_lines.append(
            (
            f"C5 D_KA: {knee_ank_dis:.3f} = min ({left_k:.3f}, {right_k:.3f}) < 0.06 "
            f"[{'YES' if c6 else 'NO'}]",
            color
            )
            )



  
            vel = pose_features.get("velocity_y", 0)

            mid_hipY_mid_shldrY = vel * frames * dt
            prev_y = shoulder_y - mid_hipY_mid_shldrY

            c_vel = vel > self.velocity_threshold
            color = COLOR_PASS if c_vel else COLOR_FAIL

            debug_lines.append(
            (
            f"C6 Vel: {vel:.1f} = Diff_Y:{mid_hipY_mid_shldrY:.1f} ({shoulder_y:.1f}-{prev_y:.1f}) > {self.velocity_threshold} "
            f"[{'YES' if c_vel else 'NO'}]",
            color
            )
            )

          



        
            c7 = nose >= self.head_ground_threshold
            color = COLOR_PASS if c7 else COLOR_FAIL

            debug_lines.append(
                (f"C7 HeadProxGrnd: {nose:.2f} >= {self.head_ground_threshold} [{'YES' if c7 else 'NO'}]", color)
            )


          
            fc = pose_features.get("fall_criteria_met", 0)

          
            if self.fc_display_snapshot is not None:
                fc = self.fc_display_snapshot




            cf = self.consecutive_fall_frames
            color_fc = COLOR_PASS if fc >= 3 else COLOR_FAIL
            color_cf = COLOR_PASS if cf >= 2 else COLOR_FAIL

            debug_lines.append(
                (f"Fall_crit_count: {fc} >= 3 [{'YES' if fc >= 3 else 'NO'}]", color_fc)
            )

            debug_lines.append(
                (f"Consecutive_fall_frames: {cf} >= 2 [{'YES' if cf >= 2 else 'NO'}]", color_cf)
            )
            

          

          

          




            


            


           


          

        output_frame = draw_text_panel(
            output_frame,
            debug_lines,
            position="bottom_left"
        )





        if self.video_fall_detected:


            text = "FALL DETECTED"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.1
            thickness = 3

            (text_width, text_height), _ = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness
            )

            x = (self.frame_width - text_width) // 2 - 70
            y = 180   

            cv2.putText(
                output_frame,
                text,
                (x, y),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA
            )

      



        self.fc_display_snapshot = None
        return output_frame, falls_detected, fall_data
    
    def _check_critical_points_ratio(self, keypoints):
     
   
        
        valid_points = 0
        total_points = 0
        
    
        critical_indices = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        
        for idx in critical_indices:
            if idx < len(keypoints) and keypoints[idx] is not None:
                visibility = keypoints[idx].visibility if hasattr(keypoints[idx], 'visibility') else 1.0
                if visibility > 0.5:
                    valid_points += 1
            total_points += 1
        
     
        return 0 if total_points == 0 else valid_points / total_points

    def track_persons(self, frame, person_boxes):
    
        current_time = time.time()
        current_boxes = person_boxes
        box_to_id = {}
        
       
        if not self.person_trackers:
            for i, box in enumerate(current_boxes):
                self.person_trackers[self.next_person_id] = {
                    'box': box,
                    'last_seen': current_time,
                    'is_fallen': False
                }
                box_to_id[i] = self.next_person_id
                self.next_person_id += 1
            return box_to_id
        
      
        matched_indices = []
        
        for i, current_box in enumerate(current_boxes):
            best_iou = 0.3  
            best_id = None
            
            for person_id, tracker_info in self.person_trackers.items():
                previous_box = tracker_info['box']
                
               
                x1_bbox_left = max(current_box[0], previous_box[0])
                y1_bbox_top = max(current_box[1], previous_box[1])
                x2_bbox_right = min(current_box[2], previous_box[2])
                y2_bbox_bottom = min(current_box[3], previous_box[3])
                
                if x2_bbox_right < x1_bbox_left or y2_bbox_bottom < y1_bbox_top:
                  
                    continue
                    
                current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                previous_area = (previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])
                intersection = (x2_bbox_right - x1_bbox_left) * (y2_bbox_bottom - y1_bbox_top)
                union = current_area + previous_area - intersection
                
                iou = intersection / union
                
                if iou > best_iou:
                    best_iou = iou
                    best_id = person_id
            
            if best_id is not None:
            
                box_to_id[i] = best_id
                self.person_trackers[best_id]['box'] = current_box
                self.person_trackers[best_id]['last_seen'] = current_time
                matched_indices.append(best_id)
            else:
              
                self.person_trackers[self.next_person_id] = {
                    'box': current_box,
                    'last_seen': current_time,
                    'is_fallen': False
                }
                box_to_id[i] = self.next_person_id
                self.next_person_id += 1
        
      
        ids_to_remove = []
        for person_id, tracker_info in self.person_trackers.items():
            if current_time - tracker_info['last_seen'] > 5.0:
                ids_to_remove.append(person_id)
               
                if person_id in self.fallen_person_ids:
                    self.fallen_person_ids.remove(person_id)
        
        for person_id in ids_to_remove:
            del self.person_trackers[person_id]
        
        return box_to_id 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from datetime import datetime
import time # Used for logging/time calculation, though logging is commented out

# =========================================================================
# 1. MEDIAPIPE INITIALIZATION (GLOBAL)
# =========================================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize Pose Detector (used globally in analyze_frame)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================================
# 2. FASTAPI APP & MIDDLEWARE
# =========================================================================
app = FastAPI(title="AI Physiotherapy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows connection from the frontend (e.g., localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# 3. EXERCISE DATA & PYDANTIC MODELS
# =========================================================================

EXERCISE_PLANS = {
    "shoulder injury": {
        "ailment": "shoulder injury",
        "exercises": [
            {"name": "Shoulder Flexion", "description": "Raise your arm forward and up", "target_reps": 12, "sets": 3, "rest_seconds": 30},
            {"name": "Shoulder Abduction", "description": "Raise your arm out to the side", "target_reps": 12, "sets": 3, "rest_seconds": 30},
            {"name": "Shoulder Pendulum", "description": "Gently swing your arm in small circles", "target_reps": 10, "sets": 3, "rest_seconds": 30}
        ],
        "difficulty_level": "beginner",
        "duration_weeks": 6
    },
    "elbow injury": {
        "ailment": "elbow injury",
        "exercises": [
            {"name": "Elbow Flexion", "description": "Bend your elbow bringing hand toward shoulder", "target_reps": 15, "sets": 3, "rest_seconds": 30},
            {"name": "Elbow Extension", "description": "Straighten your elbow completely", "target_reps": 15, "sets": 3, "rest_seconds": 30},
            {"name": "Wrist Rotation", "description": "Rotate your wrist palm up and down", "target_reps": 12, "sets": 3, "rest_seconds": 30}
        ],
        "difficulty_level": "beginner",
        "duration_weeks": 4
    },
    "wrist injury": {
        "ailment": "wrist injury",
        "exercises": [
            {"name": "Wrist Flexion", "description": "Bend your wrist forward and back", "target_reps": 15, "sets": 3, "rest_seconds": 30},
            {"name": "Wrist Extension", "description": "Extend your wrist upward", "target_reps": 15, "sets": 3, "rest_seconds": 30},
            {"name": "Wrist Circles", "description": "Make circular motions with your wrist", "target_reps": 10, "sets": 3, "rest_seconds": 30}
        ],
        "difficulty_level": "beginner",
        "duration_weeks": 3
    }
}

class AilmentRequest(BaseModel):
    ailment: str

class FrameRequest(BaseModel):
    frame: str
    exercise_name: str
    previous_state: Optional[dict] = None

class SessionResult(BaseModel):
    reps: int
    feedback: List[dict]
    accuracy_score: float
    state: dict

# =========================================================================
# 4. UTILITY FUNCTIONS (ADAPTED FROM YOUR CODE)
# =========================================================================

# The EMA smoothing function is not strictly needed here as the frontend often
# sends discontinuous frames, but we'll keep it for potential future smoothing.
def ema_smooth(value, prev, alpha=0.2):
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev

def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    # Use cross-product method for robustness
    ba = a - b
    bc = c - b
    
    # Calculate angle using normalized dot product
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def get_landmark_coords(landmarks, indices):
    """Extracts X, Y, Z coordinates for specified indices (world landmarks)."""
    return [[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices]

# --- Exercise Analysis ---

def analyze_movement(landmarks, exercise_name: str, side: str):
    """Identifies the three landmarks needed and returns the angle and feedback."""
    
    # NOTE: Since this is an API, we assume the movement target ranges 
    # are hardcoded or passed in, as calibration is complex to run in a stateless API.
    # We use hardcoded threshold angles for demonstration.
    
    feedback = []
    
    if "flexion" in exercise_name:
        # Shoulder Flexion: HIP-SHOULDER-ELBOW angle
        # Target: Angle closes as arm lifts forward (hip is the first point)
        if side == "left":
            idx = [mp_pose.PoseLandmark.LEFT_HIP.value,
                   mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                   mp_pose.PoseLandmark.LEFT_ELBOW.value]
            # Use fixed angle range (0=straight up, 180=relaxed)
            MIN_ANGLE = 20  # Max Flexion (Arm up)
            MAX_ANGLE = 160 # Resting Position
        else: # right
            idx = [mp_pose.PoseLandmark.RIGHT_HIP.value,
                   mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                   mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            MIN_ANGLE = 20
            MAX_ANGLE = 160
            
        coords = get_landmark_coords(landmarks, idx)
        angle = calculate_angle(*coords)
        
        # Simple quality check
        if angle < 140 and angle > 40:
             feedback.append({"type": "progress", "message": "Good range of motion"})
        elif angle < 40:
             feedback.append({"type": "correction", "message": "Try not to bend your elbow"})
        
        
    elif "abduction" in exercise_name:
        # Shoulder Abduction: ELBOW-SHOULDER-HIP angle
        # Target: Angle opens as arm lifts sideways (hip is the last point)
        if side == "left":
            # Note: We use SHOULDER-ELBOW-WRIST for elbow/wrist control, 
            # but for Abduction, often it's ELBOW - SHOULDER - HIP/TORSO.
            # Let's use HIP-SHOULDER-ELBOW but interpret the angle for abduction motion
            # (where arm moves away from body).
            idx = [mp_pose.PoseLandmark.LEFT_HIP.value,
                   mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                   mp_pose.PoseLandmark.LEFT_ELBOW.value]
            MIN_ANGLE = 30  # Max Abduction (Arm out)
            MAX_ANGLE = 170 # Resting Position
        else: # right
            idx = [mp_pose.PoseLandmark.RIGHT_HIP.value,
                   mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                   mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            MIN_ANGLE = 30
            MAX_ANGLE = 170
        
        coords = get_landmark_coords(landmarks, idx)
        angle = calculate_angle(*coords)
        
        if angle < 160 and angle > 40:
             feedback.append({"type": "progress", "message": "Maintain controlled movement"})
        elif angle < 40:
             feedback.append({"type": "correction", "message": "Ensure your arm is straight, raise slightly higher"})

    elif "elbow flexion" in exercise_name:
        # Elbow Flexion: SHOULDER-ELBOW-WRIST angle
        # Target: Angle closes as hand moves toward shoulder
        if side == "left":
            idx = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                   mp_pose.PoseLandmark.LEFT_ELBOW.value,
                   mp_pose.PoseLandmark.LEFT_WRIST.value]
            MIN_ANGLE = 30  # Max Flexion (Bent)
            MAX_ANGLE = 170 # Straight
        else: # right
            idx = [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                   mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                   mp_pose.PoseLandmark.RIGHT_WRIST.value]
            MIN_ANGLE = 30
            MAX_ANGLE = 170

        coords = get_landmark_coords(landmarks, idx)
        angle = calculate_angle(*coords)
        
        if angle > 150:
            feedback.append({"type": "correction", "message": "Bend your elbow more to hit max flexion."})
        elif angle < 50:
            feedback.append({"type": "encouragement", "message": "Great bend! Now straighten slowly."})
            
    else:
        # Default/unrecognized exercise
        angle = 0
        MIN_ANGLE = 0
        MAX_ANGLE = 0
        feedback = [{"type": "warning", "message": "Exercise logic not implemented yet."}]
        
    # State management needs to be done outside this function
    return angle, MIN_ANGLE, MAX_ANGLE, feedback

# =========================================================================
# 5. API ENDPOINTS (ADAPTED FOR POSE ESTIMATION)
# =========================================================================

@app.get("/")
def root():
    return {"message": "AI Physiotherapy API is running", "status": "healthy"}

@app.post("/api/get_plan")
def get_exercise_plan(request: AilmentRequest):
    """Returns the static exercise plan for a given ailment."""
    ailment = request.ailment.lower()
    
    # Calibration is not run here, but static MIN/MAX are needed for the frontend.
    # We will pass hardcoded general values for demonstration.
    
    if ailment in EXERCISE_PLANS:
        plan = EXERCISE_PLANS[ailment]
        
        # Add a dummy calibration range to the plan for frontend UI purposes
        # In a real app, this would come from a user profile/database
        for exercise in plan["exercises"]:
            name = exercise["name"].lower()
            if "flexion" in name:
                 exercise["min_angle"] = 30 # Fully contracted
                 exercise["max_angle"] = 170 # Fully extended
            elif "abduction" in name:
                 exercise["min_angle"] = 40 
                 exercise["max_angle"] = 170 
            elif "wrist" in name:
                 exercise["min_angle"] = 60
                 exercise["max_angle"] = 120
            else:
                 exercise["min_angle"] = 50
                 exercise["max_angle"] = 150
            
        return plan

    available = list(EXERCISE_PLANS.keys())
    raise HTTPException(
        status_code=404,
        detail=f"Exercise plan not found for '{ailment}'. Available plans: {available}"
    )

@app.post("/api/analyze_frame")
def analyze_frame(request: FrameRequest):
    """Processes a single Base64 image frame for pose analysis and rep counting."""
    try:
        # 1. Decode the image frame
        img_data = base64.b64decode(request.frame.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Process the frame with MediaPipe Pose
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # We must use 'pose_world_landmarks' for reliable 3D angle calculation
        results = pose.process(image_rgb)

        # 3. Check for landmarks
        if not results.pose_world_landmarks:
            return {
                "reps": request.previous_state.get("reps", 0) if request.previous_state else 0,
                "feedback": [{"type": "warning", "message": "No pose detected. Please stand in view of camera"}],
                "accuracy_score": 0.0,
                "state": request.previous_state or {"reps": 0, "stage": "down", "angle": 0}
            }

        landmarks = results.pose_world_landmarks.landmark
        exercise_name = request.exercise_name.lower()
        
        # State Initialization
        current_state = request.previous_state or {"reps": 0, "stage": "down", "angle": 0}
        reps = current_state.get("reps", 0)
        stage = current_state.get("stage", "down")
        
        # Determine side (hardcoded to left arm for simplicity, can be dynamic)
        side = "left" 
        
        # 4. Analyze movement and get angle/feedback
        angle, min_target, max_target, feedback = analyze_movement(landmarks, exercise_name, side)
        
        # Smooth angle (optional, using EMA on the client might be better)
        # smooth_angle = ema_smooth(angle, current_state.get("angle")) # Not used for logic to avoid state complexity
        
        # 5. Rep Counting Logic (Flexion/Extension)
        # Note: This uses the hardcoded min_target/max_target from analyze_movement for simplicity.
        # In a real app, these values would come from the database/initial calibration.
        
        # The down stage represents the fully extended/resting position (MAX_ANGLE)
        # The up stage represents the fully contracted/working position (MIN_ANGLE)

        if angle > max_target - 15 and stage == "up":
            # Transition from contracted (up) back to resting (down)
            stage = "down"
            reps += 1
            feedback.append({"type": "encouragement", "message": f"Rep {reps} completed! Slow down the return."})

        elif angle < min_target + 15 and stage == "down":
            # Transition from resting (down) to contracted (up)
            stage = "up"
            feedback.append({"type": "instruction", "message": "Hold contracted position for 1 second..."})


        accuracy = min(100.0, (reps / 10.0) * 100) if reps > 0 else 0.0

        # 6. Return updated state
        return {
            "reps": reps,
            "feedback": feedback,
            "accuracy_score": round(accuracy, 2),
            "state": {"reps": reps, "stage": stage, "angle": round(angle, 1)}
        }

    except Exception as e:
        # Includes an exception if landmark list doesn't have the required index
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

@app.get("/api/progress/{user_id}")
def get_progress(user_id: str):
    """Returns mock progress data (Unchanged)."""
    return {
        "user_id": user_id,
        "total_sessions": 12,
        "total_reps": 450,
        "average_accuracy": 87.5,
        "streak_days": 5,
        "weekly_data": [
            {"day": "Mon", "reps": 60, "accuracy": 85},
            {"day": "Tue", "reps": 70, "accuracy": 88},
            {"day": "Wed", "reps": 65, "accuracy": 86},
            {"day": "Thu", "reps": 75, "accuracy": 90},
            {"day": "Fri", "reps": 80, "accuracy": 89},
            {"day": "Sat", "reps": 55, "accuracy": 84},
            {"day": "Sun", "reps": 45, "accuracy": 82}
        ],
        "recent_sessions": [
            {
                "date": "2025-10-01",
                "exercise": "Shoulder Flexion",
                "reps": 12,
                "accuracy": 89
            },
            {
                "date": "2025-09-30",
                "exercise": "Elbow Flexion",
                "reps": 15,
                "accuracy": 92
            }
        ]
    }

# =========================================================================
# 6. MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    # Make sure you are using the correct executable from the environment
    # where mediapipe is installed (e.g., Python 3.10/3.12).
    uvicorn.run(app, host="0.0.0.0", port=8000)
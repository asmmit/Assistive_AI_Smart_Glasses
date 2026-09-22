import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ContextLoop:
    def __init__(self, every_n_frames=10):
        self.every_n_frames = every_n_frames
        self.frame_count = 0
        self.latest_faces = []
        
        # Ensure assets directory exists
        os.makedirs("assets", exist_ok=True)
        self.model_path = os.path.join("assets", "blaze_face_short_range.tflite")
        
        # Automatically download the MediaPipe Face Detector model if missing
        if not os.path.exists(self.model_path):
            print("[ContextLoop] Downloading MediaPipe face detector model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            try:
                urllib.request.urlretrieve(model_url, self.model_path)
                print("[ContextLoop] Model downloaded successfully.")
            except Exception as e:
                print(f"[ContextLoop] Failed to download model automatically: {e}")

        # Initialize the modern MediaPipe Tasks Face Detector
        try:
            if os.path.exists(self.model_path):
                base_options = python.BaseOptions(model_asset_path=self.model_path)
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.detector = vision.FaceDetector.create_from_options(options)
                self.mp_available = True
                print("[ContextLoop] Modern MediaPipe Tasks Face Detector initialized successfully.")
            else:
                self.mp_available = False
                print("[ContextLoop] Face detector model path not found. Face detection disabled.")
        except Exception as e:
            self.mp_available = False
            print(f"[ContextLoop] Error initializing MediaPipe Face Detector: {e}")

    def process(self, frame):
        """
        Processes a frame periodically to detect faces using the modern Tasks API.
        """
        self.frame_count += 1
        if not self.mp_available:
            return []

        # Run detection every N frames to save compute/CPU cycles
        if self.frame_count % self.every_n_frames == 0:
            try:
                # Convert OpenCV BGR image to RGB and wrap into mp.Image
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Perform detection
                result = self.detector.detect(mp_image)
                self.latest_faces = result.detections if result and result.detections else []
            except Exception as e:
                print(f"[ContextLoop Error] {e}")
                self.latest_faces = []

        return self.latest_faces

    def draw_annotations(self, frame):
        """
        Draws bounding boxes around detected faces on the video frame.
        """
        if not self.mp_available:
            return frame

        h, w, _ = frame.shape
        for detection in self.latest_faces:
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            
            # Draw rectangle around face
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(frame, "Face", (bbox.origin_x, max(0, bbox.origin_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame
    def should_run(self, has_person):
        """
        Determines whether the context loop should run based on whether a person is detected.
        """
        return self.mp_available

    def process_async(self, frame):
        """
        Wrapper to match main.py's asynchronous call expectation by delegating to process().
        """
        return self.process(frame)

    def get_latest(self):
        """
        Returns a dictionary containing the latest context information to satisfy main.py.
        """
        message = f"{len(self.latest_faces)} face(s) detected" if self.latest_faces else ""
        return {
            "faces": self.latest_faces,
            "message": message
        }
import cv2
import sys
import numpy as np
from ultralytics import YOLO

class AdvancedVehicleDetectorYOLOv8:
    def __init__(self):
        self.stream_url = "your stream URL here" # Replace with a valid stream URL
        self.setup_yolo()
        
        self.frame_count = 0
        self.total_vehicles_detected = 0
        
    def setup_yolo(self):
        try:
            print("Downloading YOLOv8 ...")
            
            # YOLOv8 model
            self.model = YOLO('../6_Deep_Learning_for_CV/deeplearning_workspace/models/yolov8n.pt')
            # Aternativies yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, 
            
            # COCO dataset vehicle classes
            self.vehicle_classes = [0, 2, 3, 5, 7]  # car, motorcycle, bus, truck
            self.class_names = {
                0: 'person',  # Person class added 
                2: 'car',
                3: 'motorcycle', 
                5: 'bus',
                7: 'truck'
            }
            
            # Color codes BGR
            self.colors = {
                'person': (255, 255, 0),  # Yellow
                'car': (0, 255, 0),        # Green
                'motorcycle': (255, 0, 0),  # Blue
                'bus': (0, 0, 255),        # Red
                'truck': (0, 255, 255)     # Orange
            }
            
            print("YOLOv8 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            print("Installation: pip install ultralytics")
            sys.exit()
    
    def preprocess_frame(self, frame):
        """ Frame normalization and resizing"""
        height, width = frame.shape[:2]
        
        if width < 480 or height < 480:
            scale = max(480/width, 480/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def detect_vehicles(self, frame):
        """YOLOv8 vechicle detection"""
        try:
            processed_frame = self.preprocess_frame(frame)
            
            # YOLOv8 ile tespit
            results = self.model(
                processed_frame,
                conf=0.25,
                iou=0.45,
                max_det=300,
                verbose=False,
                imgsz=640,
                device='cpu'
            )
            
            vehicles = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        
                        if class_id in self.vehicle_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf[0].item()
                            
                            w, h = x2 - x1, y2 - y1
                            if w > 25 and h > 25 and confidence > 0.3:
                                vehicles.append({
                                    'box': (int(x1), int(y1), int(w), int(h)),
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'class_name': self.class_names[class_id]
                                })
            
            return vehicles
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_detections(self, frame, vehicles):
        """Draw detected vehicles on the frame"""
        for vehicle in vehicles:
            x, y, w, h = vehicle['box']
            class_name = vehicle['class_name']
            confidence = vehicle['confidence']
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Main bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Reliability bar
            bar_width = int(w * confidence)
            cv2.rectangle(frame, (x, y - 20), (x + bar_width, y - 10), color, -1)
            cv2.rectangle(frame, (x, y - 20), (x + w, y - 10), color, 1)
            
            # Label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y - 10), color, -1)
            cv2.putText(frame, label, (x + 5, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Statistics
        total_vehicles = len(vehicles)
        cv2.putText(frame, f'Total: {total_vehicles}', (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Count by vehicle type
        vehicle_counts = {}
        for vehicle in vehicles:
            class_name = vehicle['class_name']
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
        
        y_offset = 80
        for vehicle_type, count in vehicle_counts.items():
            color = self.colors.get(vehicle_type, (255, 255, 255))
            cv2.putText(frame, f'{vehicle_type.capitalize()}: {count}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 35
        
        cv2.putText(frame, f'Frame: {self.frame_count}', (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        
        cap = cv2.VideoCapture(self.stream_url)
        
        if not cap.isOpened():
            print("Stream connection failed!")
            print(f"Control the URL: {self.stream_url}")
            sys.exit()
        
        
        window_name = 'YOLOv8 vehicle Detection'
        
        
        cv2.destroyAllWindows()
        
        # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 700)
        
        
        cv2.moveWindow(window_name, 100, 100)
        
        print("YOLOv8 vehicle detection launched!")
        print("Controls:")
        print("- 'q'  or ESC : quit")
        print("- 's' : Screenshot")
        
        screenshot_count = 0
        window_closed = False
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Video stream closed. Terminating program...")
                break
            
            self.frame_count += 1
            
            vehicles = self.detect_vehicles(frame)
            frame = self.draw_detections(frame, vehicles)

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    if not window_closed:
                        print("The window has been closed. The program is terminating...")
                        window_closed = True
                        break
                
                cv2.imshow(window_name, frame)
                
            except cv2.error:
                print("Windows error. Terminating program...")
                break
            

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' veya ESC
                print("The user is logged out.")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'vehicle_detection_{screenshot_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Screenshot: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"The program was terminated. A total of {self.frame_count} frames were processed.")

if __name__ == "__main__":
    try:
        detector = AdvancedVehicleDetectorYOLOv8()
        detector.run()
    except KeyboardInterrupt:
        print("\nThe program was stopped by the user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:

        cv2.destroyAllWindows()
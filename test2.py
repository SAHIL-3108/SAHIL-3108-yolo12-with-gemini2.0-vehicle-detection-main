import cv2
import numpy as np
import os
import time
import base64
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ultralytics import YOLO
import cvzone

# --- CONFIGURATION ---
GOOGLE_API_KEY = "AIzaSyB-5PG_grE35Pyg7mDo9GMTgUsLY2RZj2E"
ESP32_URL = "http://192.168.153.216/data"  # Make sure this matches your ESP32 endpoint
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


class VehicleDetectionProcessor:
    def __init__(self, video_file, vehicle_model_path="yolov5x.pt", plate_model_path="yolov5x.pt"):
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.names = self.vehicle_model.names

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise FileNotFoundError("Error: Could not open video file.")

        self.areas = [np.array([[100, 100], [900, 100], [900, 200], [100, 200]], np.int32)]
        self.processed_track_ids = set()
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.current_date = time.strftime("%Y-%m-%d")
        self.output_filename = f"vehicle_data_{self.current_date}.txt"
        self.cropped_images_folder = "cropped_vehicles"
        os.makedirs(self.cropped_images_folder, exist_ok=True)

        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Track ID | Vehicle Type | Vehicle Color | Vehicle Company | Plate\n")
                file.write("-" * 80 + "\n")

    def analyze_image_with_gemini(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
            message = HumanMessage(content=[
                {"type": "text", "text": "Analyze this image and extract only the following details:\n\n|Vehicle Type(Name of Vehicle) | Vehicle Color | Vehicle Company | Plate |"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "description": "Detected vehicle"}
            ])
            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Error invoking Gemini model: {e}")
            return "Error processing image."

    def send_to_esp32_with_retry(self, payload, retries=3, delay=1):
        for attempt in range(retries):
            try:
                response = requests.post(ESP32_URL, json=payload, timeout=2)
                if response.ok:
                    print("✅ ESP32: Data sent.")
                    return True
                else:
                    print(f"⚠️ ESP32 Error [{response.status_code}]: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
        print("❌ All attempts to send data failed.")
        return False

    def process_crop_image(self, image, track_id):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = os.path.join(self.cropped_images_folder, f"vehicle_{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_filename, image)
        response_content = self.analyze_image_with_gemini(image_filename)
        extracted_data = response_content.split("\n")[2:]

        if extracted_data:
            with open(self.output_filename, "a", encoding="utf-8") as file:
                for row in extracted_data:
                    if "--------------" in row or not row.strip():
                        continue
                    values = [col.strip() for col in row.split("|")[1:-1]]
                    if len(values) == 4:
                        vehicle_type, vehicle_color, vehicle_company, plate = values
                        data_line = f"{timestamp} | Track ID: {track_id} | {vehicle_type} | {vehicle_color} | {vehicle_company} | {plate}\n"
                        file.write(data_line)
                        print(f"✅ Data saved for track ID {track_id}.")

                        payload = {
                            "timestamp": timestamp,
                            "track_id": track_id,
                            "vehicle_type": vehicle_type,
                            "vehicle_color": vehicle_color,
                            "vehicle_company": vehicle_company,
                            "plate": plate
                        }
                        self.send_to_esp32_with_retry(payload)

    def crop_and_process(self, frame, box, track_id):
        if track_id in self.processed_track_ids:
            return
        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        self.processed_track_ids.add(track_id)
        self.executor.submit(self.process_crop_image, cropped_image, track_id)

    def process_video_frame(self, frame):
        frame = cv2.resize(frame, (1020, 600))
        results = self.vehicle_model.track(frame, persist=True)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)
            allowed_classes = ["car", "truck"]

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                class_name = self.names[class_id]
                if class_name not in allowed_classes:
                    continue
                x1, y1, x2, y2 = map(int, box)
                if any(cv2.pointPolygonTest(area, (x2, y2), False) >= 0 for area in self.areas):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, class_name, (x1, y1), 1, 1)
                    self.crop_and_process(frame, box, track_id)

        for area in self.areas:
            cv2.polylines(frame, [area], True, (0, 255, 0), 2)
        return frame

    def start_processing(self):
        cv2.namedWindow("Vehicle Detection")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.process_video_frame(frame)
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Data saved to {self.output_filename}")


if __name__ == "__main__":
    video_file = r"C:\Users\hp\Desktop\sem-6 project AI\yolo12-with-gemini2.0-vehicle-detection-main\tc (1).mp4"  # Replace with your actual path
    processor = VehicleDetectionProcessor(video_file)
    processor.start_processing()

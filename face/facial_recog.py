import cv2
import face_recognition
import numpy as np
import os
import pickle

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []

        self.db_dir = "face_database"
        self.db_file = os.path.join(self.db_dir, "faces.pkl")
        os.makedirs(self.db_dir, exist_ok=True)

        self.load_faces()

        self.registration_mode = False
        self.current_name = ""

        self.use_low_res = True  # Toggle resolution mode

    def load_faces(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data['encodings']
                    self.known_names = data['names']
                print(f"Loaded {len(self.known_names)} faces: {', '.join(self.known_names)}")
            except Exception as e:
                print(f"Error loading face database: {e}")

    def save_faces(self):
        try:
            with open(self.db_file, 'wb') as f:
                data = {
                    'encodings': self.known_faces,
                    'names': self.known_names
                }
                pickle.dump(data, f)
            print(f"Saved {len(self.known_names)} faces to database")
        except Exception as e:
            print(f"Error saving face database: {e}")

    def register_face(self, frame, name):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            return False, "No face detected"

        if len(face_locations) > 1:
            return False, "Multiple faces detected. Please ensure only one face is in the frame."

        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

        if self.known_faces:
            distances = face_recognition.face_distance(self.known_faces, face_encoding)
            if np.any(distances < 0.45):
                existing_name = self.known_names[np.argmin(distances)]
                return False, f"This face is already registered as {existing_name}"

        if name in self.known_names:
            return False, f"Name '{name}' is already in use"

        self.known_faces.append(face_encoding)
        self.known_names.append(name)
        self.save_faces()

        return True, f"Successfully registered {name}"

    def run(self):
        cap = cv2.VideoCapture(0)
        info_text = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            scale_factor = 0.3 if self.use_low_res else 1.0
            scaled_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            display_frame = frame.copy()

            if self.registration_mode:
                cv2.putText(display_frame, f"Registering: {self.current_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'r' to capture", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to cancel", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"
                    if self.known_faces:
                        distances = face_recognition.face_distance(self.known_faces, face_encoding)
                        best_match_index = np.argmin(distances)
                        if distances[best_match_index] < 0.45:
                            name = self.known_names[best_match_index]

                    # Scale back to original resolution
                    top = int(top / scale_factor)
                    right = int(right / scale_factor)
                    bottom = int(bottom / scale_factor)
                    left = int(left / scale_factor)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, name, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                cv2.putText(display_frame, "Press 'n' to register new face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'l' to toggle resolution", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'q' to quit", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if info_text:
                cv2.putText(display_frame, info_text, (10, display_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Face Recognition", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and not self.registration_mode:
                self.registration_mode = True
                self.current_name = self.prompt_for_name()
                info_text = f"Enter registration mode for {self.current_name}"
            elif key == ord('r') and self.registration_mode:
                success, message = self.register_face(frame, self.current_name)
                info_text = message
                if success:
                    self.registration_mode = False
                    cv2.putText(display_frame, message, (10, display_frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Face Recognition", display_frame)
                    cv2.waitKey(2000)
            elif key == ord('c') and self.registration_mode:
                self.registration_mode = False
                info_text = "Registration cancelled"
            elif key == ord('l'):
                self.use_low_res = not self.use_low_res
                info_text = "Low resolution ON" if self.use_low_res else "High resolution ON"

        cap.release()
        cv2.destroyAllWindows()

    def prompt_for_name(self):
        name = ""
        input_prompt = np.zeros((200, 600, 3), np.uint8)

        while True:
            input_prompt.fill(0)
            cv2.putText(input_prompt, "Enter name (press Enter when done):", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(input_prompt, name + "|", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Name Input", input_prompt)
            key = cv2.waitKey(0) & 0xFF

            if key == 13:  # Enter
                break
            elif key == 8:  # Backspace
                name = name[:-1]
            elif 32 <= key <= 126:
                name += chr(key)

        cv2.destroyWindow("Name Input")
        return name

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()

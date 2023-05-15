import sqlite3
import face_recognition
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog

class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                                id INTEGER PRIMARY KEY,
                                name TEXT NOT NULL,
                                encoding BLOB NOT NULL)''')
        self.conn.commit()

    def insert_user(self, name, encoding):
        self.cursor.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, encoding))
        self.conn.commit()

    def get_all_users(self):
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

class FaceRecognitionSystem:
    def __init__(self, master, db):
        self.master = master
        self.db = db
        self.master.title("Face Recognition System")
        self.create_widgets()

    def create_widgets(self):
        self.add_user_button = Button(self.master, text="Add User", command=self.add_user)
        self.add_user_button.pack()

        self.recognize_faces_button = Button(self.master, text="Recognize Faces", command=self.recognize_faces)
        self.recognize_faces_button.pack()

    def add_user(self):
        name = input("Enter the user's name: ")
        image_path = filedialog.askopenfilename(title="Select the user's image")

        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]

        self.db.insert_user(name, np.array(encoding).tobytes())

    def recognize_faces(self):
        known_users = self.db.get_all_users()
        known_encodings = [np.frombuffer(user[2], dtype=np.float64) for user in known_users]
        known_names = [user[1] for user in known_users]

        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    db = Database("face_recognition.db")
    root = Tk()
    app = FaceRecognitionSystem(root, db)
    root.mainloop()
    db.close()

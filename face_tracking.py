import cv2

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open video file or webcam (0 = webcam, or replace with "video.mp4")
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to grayscale (better for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Face Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video and close windows
video_capture.release()
cv2.destroyAllWindows()

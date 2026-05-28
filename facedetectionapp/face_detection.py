# Import OpenCV library
import cv2

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam capture
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Infinite loop to continuously capture frames
while True:

    # Read frame from webcam
    ret, frame = cap.read()

    # If frame not captured properly
    if not ret:
        print("Failed to capture image")
        break

    # Convert color image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangle around each detected face
    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Show output window
    cv2.imshow("Face Detection App", frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

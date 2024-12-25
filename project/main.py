import cv2
import cvzone
import os

print(cv2.__version__)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    cap.release()
    exit()

# Load the cascade classifier
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print(f"Error: Cascade file '{cascade_path}' not found.")
    cap.release()
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the overlay image
overlay_path = 'pirate.png'
if not os.path.exists(overlay_path):
    print(f"Error: Overlay file '{overlay_path}' not found.")
    cap.release()
    exit()

overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

def resize_overlay(overlay, face_width, face_height):
    # Calculate aspect ratio of overlay
    overlay_height, overlay_width = overlay.shape[:2]
    aspect_ratio = overlay_width / overlay_height
    
    # Calculate new dimensions maintaining aspect ratio
    new_width = int(face_width * 1.2)  # Make overlay slightly wider than face
    new_height = int(new_width / aspect_ratio)
    
    return cv2.resize(overlay, (new_width, new_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Resize overlay to match face dimensions
        overlay_resized = resize_overlay(overlay, w, h)
        
        # Calculate position to center overlay on face
        overlay_h, overlay_w = overlay_resized.shape[:2]
        x_offset = x - (overlay_w - w) // 2
        y_offset = y - (overlay_h - h) // 2
        
        # Apply overlay
        frame = cvzone.overlayPNG(frame, overlay_resized, [x_offset, y_offset])

    cv2.imshow("Snap Filter", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

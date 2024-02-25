import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')  # Make sure to provide the correct path to the cascade classifier XML file

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the detected area

    # Display the frame
    cv2.imshow('Body Detection', frame)

    if cv2.waitKey(1) == 32:  # 32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()


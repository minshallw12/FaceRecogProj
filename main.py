import threading

import cv2
from deepface import DeepFace

# Capture video
cap = cv2.VideoCapture(0)

#set proportions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#counts which frame
counter = 0

#supplied image to compare each frame to
reference_img = cv2.imread("reference2.jpg")  # use your own image here

#global var
face_match = False

# Checks the frame against the face
def check_face(frame):

    global face_match
    try:
        #verifies frame against reference image
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

# main program loop
while True:
    #captures any return value and the frame from camera
    ret, frame = cap.read()

    # if there is a return value
    if ret:
        # call check_face function every x frames
        if counter % 10 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        # increment frame counter
        counter += 1

        # Do something if the face matches or doesnt match
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # show the video
        cv2.imshow('video', frame)

    # press 'q' to quit to break main loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# destroy all when program ends
cv2.destroyAllWindows()
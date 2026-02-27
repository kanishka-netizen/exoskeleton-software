import cv2   #for image, video related operations
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

#getting base model/initial set-up
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)

#importing video
cap = cv2.VideoCapture("wrist1.mp4")

if not cap.isOpened():
    print("Video not opening.")
    exit()

timestamp = 0
previous_angle = None
unwrapped_angle = 0

#storing extreme angles
min_angle = float("inf")
max_angle = float("-inf")

#processing the video imported earlier
while cap.isOpened():
    ret, frame = cap.read() #frame stores the image at that specific point and ret stores if the frame is actually being read or not. so when ret becomes false, the video is closed
    if not ret:
        break

    timestamp += 1  # must strictly increase

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting blue->green->red to red->green->blue, if we don't do this detection wont be very accurate

    mp_image = mp.Image( #converting raw numpy array to mediapipe expectant image format
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect_for_video(mp_image, timestamp)  #detect_for_video mode for imported video

    if result.hand_landmarks:
        h, w, _ = frame.shape

        for hand_landmarks in result.hand_landmarks:

            wrist = hand_landmarks[0]  #0 for wrist point
            wx = int(wrist.x * w)
            wy = int(wrist.y * h)

            middle = hand_landmarks[9]  #9 for middle base knuckle (MCP)
            mx = int(middle.x * w)
            my = int(middle.y * h)

            cv2.circle(frame, (wx, wy), 8, (0,255,0), -1)   #drawing point on wrist
            cv2.circle(frame, (mx, my), 8, (255,0,0), -1)   #drawing point on MCP
            cv2.line(frame, (wx, wy), (mx, my), (255,255,255), 2)  #drawing axis line

            #computing angle
            dx = mx - wx
            dy = my - wy

            raw_angle = math.degrees(math.atan2(dy, dx))

            if previous_angle is None:
                unwrapped_angle = raw_angle
            else:
                delta = raw_angle - previous_angle

                # Fix wrap-around jump
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360

                unwrapped_angle += delta

            previous_angle = raw_angle
            angle = unwrapped_angle

            #updating extremes
            if angle < min_angle:
                min_angle = angle

            if angle > max_angle:
                max_angle = angle

            #displaying live angles
            cv2.putText(frame, f"Angle: {angle:.2f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,255), 2)

    cv2.imshow("Wrist ROM Tracking", frame)
    #cv2.imshow("Wrist ROM Tracking")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#final results.
if min_angle != float("inf") and max_angle != float("-inf"):
    rom = max_angle - min_angle
    print("Minimum angle:", min_angle)
    print("Maximum angle:", max_angle)
    print("Range of Motion (ROM):", rom, "degrees")
else:
    print("Not enough movement detected.")
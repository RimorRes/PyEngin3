import numpy as np
import cv2

screen_width = 2560
screen_height = 1440

cap = cv2.VideoCapture(1)

# Check if the cameras are opened correctly
if not cap.isOpened():
    raise IOError('Cannot access stereo cameras.')

# Resolution and rate of the camera frames (both L and R views stitched side by side)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

mid = frame_width // 2
x = max((frame_width - screen_width) // 2, 0)
y = max((frame_height - screen_height) // 2, 0)

print(frame_width, 'x', frame_height, '@', fps)

def grab():
    ret, frame = cap.read()

    # (x, y)
    if ret:
        img_l, img_r = frame[:, :mid, :], frame[:, mid:, :]
        img_l = img_l[y:y+screen_height, x:x+screen_width, :]
        img_r = img_r[y:y+screen_height, x:x+screen_width, :]
        res = np.concatenate((img_l, img_r), axis=1)

        return res


while True:
    render = grab()
    # cv2.imshow('Stereo Cam', cv2.resize(render, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.imshow('Stereo Cam', render)

    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        break

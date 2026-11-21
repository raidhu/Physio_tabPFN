import cv2
from physio_tabPFN import CameraFeed

with CameraFeed(source=0, width=1280, height=720) as cam:
    for rgb_frame in cam.frames():
        # rgb_frame is now ready for MediaPipe / PyTorch / YOLO

        # Note: cv2.imshow expects BGR, so convert back ONLY for preview display
        cv2.imshow("Physio Camera Feed (Press 'q' to quit)", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
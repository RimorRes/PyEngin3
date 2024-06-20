import numpy as np
import glob
import cv2

# TODO: add logging, warnings


class StereoCam:

    def __init__(self, cam_id):

        self.cap = cv2.VideoCapture(cam_id)

        # Check if the cameras are opened correctly
        if not self.cap.isOpened():
            raise IOError('Cannot access stereo cameras.')

        # Resolution and rate of the camera frames (both L and R views stitched side by side)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Map init for undistorting
        self.undistort = False
        self.map1x, self.map1y = None, None
        self.map2x, self.map2y = None, None

    def calibration_wizard(self, output_path, chessboard_size=(9, 6), square_size_mm=20):
        w, h = self.frame_width // 2, self.frame_height

        # Take snapshots
        num = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            key = cv2.waitKey(1)

            if key == 27:
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                cv2.imwrite('cal' + str(num) + '.png', frame)
                print('Image saved!')
                num += 1

            cv2.imshow('Stereo Cam', cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        objp = objp * square_size_mm

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points1 = []  # 2d points in image plane for cam 1.
        img_points2 = []  # 2d points in image plane for cam 2.

        images = glob.glob('*.png')  # TODO: clean-up file creation
        for fpath in images:
            frame = cv2.imread(fpath)
            img_1, img_2 = self.cut(frame)

            gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_1, corners_1 = cv2.findChessboardCorners(gray_1, chessboard_size, None)
            ret_2, corners_2 = cv2.findChessboardCorners(gray_2, chessboard_size, None)

            # If found, add object points, image points (after refining them)
            if ret_2 and ret_2:
                print(fpath, 'is valid')
                obj_points.append(objp)
                refine_corners_1 = cv2.cornerSubPix(gray_1, corners_1, (11, 11), (-1, -1), criteria)
                img_points1.append(refine_corners_1)
                refine_corners_2 = cv2.cornerSubPix(gray_2, corners_2, (11, 11), (-1, -1), criteria)
                img_points2.append(refine_corners_2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img_1, chessboard_size, refine_corners_1, ret_1)
                cv2.imshow('Image 1', img_1)
                cv2.drawChessboardCorners(img_2, chessboard_size, refine_corners_2, ret_2)
                cv2.imshow('Image 2', img_2)
                cv2.waitKey(1000)

        cv2.destroyAllWindows()

        ret, mtx1, dist1, mtx2, dist2, r, t, e, f = cv2.stereoCalibrate(
            obj_points,
            img_points1,
            img_points2,
            None,
            None,
            None,
            None,
            (w, h),
            None,
            None
        )

        if not ret:
            raise RuntimeError('Stereo camera calibration failed.')
        else:
            np.savez(output_path, mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2, r=r, t=t)

    def load_calibration(self, fpath):
        data = np.load(fpath)
        mtx1 = data['mtx1']
        dist1 = data['dist1']
        mtx2 = data['mtx2']
        dist2 = data['dist2']
        r = data['r']
        t = data['t']

        w, h = self.frame_width // 2, self.frame_height

        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, (w, h), r, t)

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(mtx1, dist1, r1, p1, (w, h), 5)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(mtx2, dist2, r2, p2, (w, h), 5)

    def __del__(self):
        self.cap.release()

    def close(self):
        self.cap.release()

    def cut(self, full_frame):
        """
        Cuts captured frame into both left and right views
        :param full_frame: captured frame (side-by-side stitched views)
        :return: img_l, img_r
        """
        mid = self.frame_width // 2
        img_l, img_r = full_frame[:, :mid, :], full_frame[:, mid:, :]

        return img_l, img_r

    def grab(self):
        ret, frame = self.cap.read()

        if ret:  # if a frame is captured
            # Split both right and left images
            img_l, img_r = self.cut(frame)

            if self.undistort:  # TODO: see if cv2.fisheye rectification works better
                img_l = cv2.remap(img_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
                img_r = cv2.remap(img_r, self.map2x, self.map2y, cv2.INTER_LINEAR)

            return img_l, img_r

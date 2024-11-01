import cv2
import numpy as np



cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)

def decode(frame):
    left = np.zeros((800,1264,3), np.uint8)
    right = np.zeros((800,1264,3), np.uint8)
    
    for i in range(800):
        left[i] = frame[i, 64: 1280 + 48] 
        right[i] = frame[i, 1280 + 48: 1280 + 48 + 1264] 
    
    return (left, right)

chessboard_size = (9,6)

# Define the criteria for termination of the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints_left = [] # 2d points in image plane for left camera
imgpoints_right = [] # 2d points in image plane for right camera

detected_chessboards = 0
while detected_chessboards<80:
    ret, frame = cap.read()
    img_left, img_right = decode(frame)
    if not ret:
        break
    

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
    
    # If found, add object points, image points (after refining them)
    if ret_left and ret_right:
        objpoints.append(objp)
        
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
        imgpoints_left.append(corners2_left)
        
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
        imgpoints_right.append(corners2_right)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img_left, chessboard_size, corners2_left, ret_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners2_right, ret_right)
        
        cv2.imshow('img_left', img_left)
        cv2.imshow('img_right', img_right)
        
        cv2.waitKey(500)
        detected_chessboards = detected_chessboards + 1
        print("chessboard detected ", detected_chessboards)
    else:
        print("no chessboard detected")


cv2.destroyAllWindows()

# Calibrate each camera individually
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# Stereo calibration
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, 
    mtx_left, dist_left, mtx_right, dist_right, 
    gray_left.shape[::-1], criteria, flags)

# Save the calibration results
np.savez('stereo_calib.npz', mtx_left=mtx_left, dist_left=dist_left, mtx_right=mtx_right, dist_right=dist_right, R=R, T=T, E=E, F=F)

# Print the calibration results
print("Left camera matrix:\n", mtx_left)
print("Left camera distortion coefficients:\n", dist_left)
print("Right camera matrix:\n", mtx_right)
print("Right camera distortion coefficients:\n", dist_right)
print("Rotation matrix between the cameras:\n", R)
print("Translation vector between the cameras:\n", T)
print("Essential matrix:\n", E)
print("Fundamental matrix:\n", F)

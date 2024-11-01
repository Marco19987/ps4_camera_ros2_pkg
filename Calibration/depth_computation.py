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


# Load the stereo calibration parameters
calib_data = np.load('stereo_calib.npz')
mtx_left = calib_data['mtx_left']
dist_left = calib_data['dist_left']
mtx_right = calib_data['mtx_right']
dist_right = calib_data['dist_right']
R = calib_data['R']
T = calib_data['T']



h,w = (800, 1264)



# Rectify the images
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, (w, h), R, T)
map_left_x, map_left_y = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, (w, h), cv2.CV_32FC1)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, (w, h), cv2.CV_32FC1)

# Create the stereo block matcher
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=128)


#Stereo matcher settings
win_size = 5
min_disp = 10
max_disp = 16 * 2 + 10
num_disp = max_disp - min_disp # Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
numDisparities = num_disp,
blockSize = 5,
uniquenessRatio = 10,
speckleWindowSize = 1000,
speckleRange = 10,
disp12MaxDiff = 25,
P1 = 8*3*win_size**2,#8*3*win_size**2,
P2 =32*3*win_size**2) #32*3*win_size**2)


while True:
    ret, frame = cap.read()
    frame_left, frame_right = decode(frame)
    
    if not ret:
        break
    
    # Rectify the images
    rectified_left = cv2.remap(frame_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(frame_right, map_right_x, map_right_y, cv2.INTER_LINEAR)
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
    
    # Compute the disparity map
    disparity = stereo.compute(gray_left, gray_right)
    
    
    # Normalize the disparity map for visualization
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)
    
    # Compute the depth map
    depth_map = cv2.reprojectImageTo3D(disparity, Q)
    
    # Display the images
    cv2.imshow('Left', rectified_left)
    cv2.imshow('Right', rectified_right)
    cv2.imshow('Disparity', disparity)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    




cv2.destroyAllWindows()
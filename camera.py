import numpy as np
import cv2 as cv
import time

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
# successes = 0
# while (successes < 10):
#     ret, img = cap.read()
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (7,6), None)
#     # If found, add object points, image points (after refining them)
#     print(f"Looking for chess pattern... ({successes}/10)")

#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners)

#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (7,6), corners2, ret)
#         cv.imshow('img', img)
#         print("Press b for selection, else press any key", successes)
#         key = cv.waitKey()
#         if key & 0xFF == ord('b'):
#             successes += 1
#         else: print("Skipped")
#         cv.destroyAllWindows()
#         start = time.time()
#         while time.time() < start + 5:
#             cap.read() # Flush camera buffer
        
# cv.destroyAllWindows()


# print("Done")


#ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)

# np.savetxt('mtx.csv', mtx, delimiter=',')
# np.savetxt('dist.csv', dist, delimiter=',')

mtx = np.loadtxt('mtx.csv', delimiter=',')
dist = np.loadtxt('dist.csv', delimiter=',')

ret, img = cap.read()

cv.imshow("a", img)
cv.waitKey(-1)

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult.png', dst)
cv.waitKey()


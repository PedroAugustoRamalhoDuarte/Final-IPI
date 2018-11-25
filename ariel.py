import cv2 as cv
import numpy as np

cap = cv.VideoCapture('overpass.mp4')

_, frame1 = cap.read()
B, G, R = cv.split(frame1)
frame1 = (B + G + R) / 3
frame1 = cv.normalize(frame1, 0, 1, norm_type=cv.NORM_MINMAX)
Bg = frame1
Thresh = np.zeros((frame1.shape[0], frame1.shape[1]))
Thresh += 0.5
alpha = 0.8

_, frame2 = cap.read()
B, G, R = cv.split(frame2)
frame2 = (B + G + R) / 3
frame2 = cv.normalize(frame2, 0, 1, norm_type=cv.NORM_MINMAX)
_, frame3 = cap.read()
B, G, R = cv.split(frame3)
frame3 = (B + G + R) / 3
frame3 = cv.normalize(frame3, 0, 1, norm_type=cv.NORM_MINMAX)

while 1:



    diff1 = np.abs(frame2 - frame3)
    diff2 = np.abs(frame1 - frame3)

    # _, diff1_T = cv.threshold(diff1, Thresh, 255, cv.THRESH_BINARY)
    # _, diff2_T = cv.threshold(diff2, Thresh, 255, cv.THRESH_BINARY)

    diff1_T = diff1 > Thresh
    diff1_T = diff1_T * diff1
    diff2_T = diff2 > Thresh
    diff2_T = diff2_T * diff2

    moving = cv.bitwise_and(diff1_T, diff2_T)
    norm_moving = cv.normalize(moving, 0, 1, norm_type=cv.NORM_MINMAX)

    Thresh1 = Thresh * norm_moving
    Thresh2 = alpha * Thresh * (1 - norm_moving) + (1 - alpha) * (5 * np.abs(frame3 - Bg)) * (1 - norm_moving)
    Thresh = Thresh1 + Thresh2

    Bg1 = Bg * norm_moving
    Bg2 = alpha * Bg * (1 - norm_moving) + (1 - alpha) * frame3 * (1 - norm_moving)
    Bg = Bg1 + Bg2

    cv.imshow('bg', Bg)
    cv.imshow('vi', frame3)
    # cv.imshow('sal1', diff1_T)
    # cv.imshow('sal2', diff2_T)
    # cv.imshow('sal3', norm_moving)
    cv.waitKey()

    frame1 = frame2
    frame2 = frame3
    _, frame3 = cap.read()
    B, G, R = cv.split(frame3)
    frame3 = (B + G + R) / 3
    frame3 = cv.normalize(frame3, 0, 1, norm_type=cv.NORM_MINMAX)
import cv2 as cv
import numpy as np


def binarizar(matriz):
    T = 160
    linhas, colunas = matriz.shape
    for i in range(0, linhas):
        for j in range(0, colunas):
            if matriz[i][j] > T:
                matriz[i][j] = 255
            else:
                matriz[i][j] = 0


def frame_difference(previus_frame, frame):
    return frame - previus_frame


def func():
    background = frame_difference(previus_frame, frame)
    a, th3 = cv.threshold(frame, 170, 255, cv.THRESH_BINARY)
    th3 = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # binarizar(background)
    cv.imshow("aloha", th3)
cap = cv.VideoCapture("teste.avi")

fgbg = cv.createBackgroundSubtractorMOG2()
cont = 0
frameanterior = 0
while(1):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
    frame = frame[:, :, 0]
    cont += 1
    if 1400 < cont < 1640:
        # -------------------------------------
        # func()

        # -------------------------------------

        fgmask = fgbg.apply(frame)
        cv.imshow('frameee', frame)
        cv.imshow('frame', fgmask)
        k = cv.waitKey(0) & 0xff
        print(cont)
        if k == 27:
            break
    previus_frame = frame
cap.release()
cv.destroyAllWindows()


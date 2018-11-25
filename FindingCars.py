import cv2 as cv
import numpy as np
from fuction import *
from background import *
cap = cv.VideoCapture("overpass.mp4")

cont = 0
previus_frame = np.zeros((720, 1280), dtype=np.uint8)
background = np.zeros((720, 1280), dtype=np.uint8)
background2 = np.zeros((720, 1280), dtype=np.uint8)
previus_background = np.zeros((720, 1280), dtype=np.uint8)
previus_foreground = np.zeros((720, 1280), dtype=np.uint8)
ground = np.zeros((720, 1280), dtype=np.uint8)
matriz = np.zeros((720, 1280), dtype=np.uint8)
ret = True
while ret:
    ret, frame_bgr = cap.read()

    # 1 Passo- Converter o frame em cinza
    frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    #imgprint("frame cinza", frame_gray)

    # 2 Passo - Diferenca do pixel atual e o anterior
    diff = cv.absdiff(frame_gray, previus_frame)
    #imgprint("diferrence", diff)

    # 3 Passo - Thresh Holder
    T = int(0.1 * np.amax(diff))
    a, binary_mask = cv.threshold(diff, T, 255, cv.THRESH_BINARY)
    #imgprint("mascaa binaria", binary_mask)

    # 4 Passo - Aplicando o algoritmo de detectar bordas
    edges = cv.Canny(binary_mask, 0, 255)
    #imgprint("bordas", edges)

    # 5 Passo - Melhorando bordas
    edges = morph_dilatation(edges)
    T = int(0.1 * np.amax(diff))
    a, edges = cv.threshold(edges, T, 1, cv.THRESH_BINARY)
    # imgprint("bordas melhoradas", edges)

    # Calculando o background e foreground da primeira segmentação
    background = frame_gray * (1 - edges)
    foreground = frame_gray * edges
    # Imprimindo Resultados
    imgprint("background", background)
    imgprint("forregound", foreground)

    # Fine-grained segmentation of Current image
    diff2 = cv.absdiff(foreground, previus_foreground)
    imgprint("diff2", diff2)
    a, diff2 = cv.threshold(diff2, 120, 255, cv.THRESH_BINARY)
    imgprint("melhorado", diff2)
    back1 = frame_gray * diff2
    fore1 = frame_gray * (1 - diff2)
    # imgprint("Back melhor", back1)
    # imgprint("Fore melhor", fore1)

    # Self adaptative background
    alfa = 0.5

    matriz[:, :] = (1 - alfa) * background[:, :] + alfa * foreground[:, :]
    imgprint("matriz", matriz)
    background2 = self_background(binary_mask, matriz, background)
    background2 = np.asarray(background2)
    imgprint("background2", background2)

    previus_frame = frame_gray[:, :]
    previus_background = background[:, :]
    previus_foreground = foreground[:, :]
cap.release()
cv.destroyAllWindows()

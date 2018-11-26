import cv2 as cv
import numpy as np
from fuction import *
from background import *
cap = cv.VideoCapture("overpass.mp4")
ret, frame = cap.read()
linhas, colunas, tipo = frame.shape

cont = 0
previus_frame = np.zeros((linhas, colunas), dtype=np.uint8)
background = np.zeros((linhas, colunas), dtype=np.uint8)
background2 = np.zeros((linhas, colunas), dtype=np.uint8)
previus_background = np.zeros((linhas, colunas), dtype=np.uint8)
previus_background2 = np.zeros((linhas, colunas), dtype=np.uint8)
previus_foreground = np.zeros((linhas, colunas), dtype=np.uint8)
ground = np.zeros((linhas, colunas), dtype=np.uint8)
matriz = np.zeros((linhas, colunas), dtype=np.uint8)
ret = True
while ret:
    ret, frame_bgr = cap.read()
    if cont == 0:
        previus_background = frame_gray

    # 1 Passo- Converter o frame em cinza
    frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    # imgprint("frame cinza", frame_gray)

    # 2 Passo - Diferenca do pixel atual e o anterior
    diff = cv.absdiff(frame_gray, previus_frame)
    # imgprint("diferrence", diff)

    # 3 Passo - Thresh Holder
    T = int(0.2* np.amax(diff))
    a, binary_mask = cv.threshold(diff, T, 255, cv.THRESH_BINARY)
    # imgprint("mascara binaria", binary_mask)

    # 4 Passo - Aplicando o algoritmo de detectar bordas
    edges = cv.Canny(binary_mask, 0, 255)
    # imgprint("bordas", edges)

    # 5 Passo - Melhorando bordas
    edges = morph_dilatation(edges)
    a, edges = cv.threshold(edges, 0, 1, cv.THRESH_BINARY)
    # imgprint("bordas melhoradas", edges * 255)

    # Calculando o background e foreground da primeira segmentação
    background = frame_gray * (1 - edges)
    foreground = frame_gray * edges
    # Imprimindo Resultados
    # imgprint("background", background)
    imgprint("forregound", foreground)

    # Fine-grained segmentation of Current image
    a, mask = cv.threshold(previus_background, T, 1, cv.THRESH_BINARY_INV)
    diff2 = cv.absdiff(foreground, previus_foreground * mask)
    # imgprint("diff2", diff2)
    a, diff2 = cv.threshold(diff2, T, 1, cv.THRESH_BINARY)
    # imgprint("melhorado", diff2)
    background2 = frame_gray * (1 - diff2)
    foreground2 = frame_gray * diff2
    #imgprint("Back melhor", background2)
    imgprint("Fore melhor", foreground2)

    # Self adaptative background
    alfa = 0.8
    matriz[:, :] = alfa * previus_background[:, :] + (1 - alfa) * frame_gray[:, :]
    # imgprint("matriz", matriz)
    # imgprint("edges antes" , edges)
    self_adap_background = self_background(diff, matriz, previus_background)
    background = np.asarray(self_adap_background)
    #imgprint("background2", background)

    # Previus
    cont += 1
    previus_frame = frame_gray[:, :]
    previus_background = background[:, :]
    previus_foreground = foreground[:, :]
    cv.waitKey(0)

cap.release()
cv.destroyAllWindows()

import cv2 as cv
import numpy as np

n = 0.5


def imgprint(name, img):
    cv.imshow(name, img)


def alfa_iluminacao(frame, previus):
    """
    Calcula a parcela do alfa que varia de acordo a luminosidade
    :param frame: Frame atual
    :param previus: Frame anterios
    :return: Parcela do alfa que varia de acordo com a luminosidade
    """

    mean1 = np.mean(frame)
    mean2 = np.mean(previus)
    max_mean = max(mean1, mean2)
    alfa = n * (1 - abs(mean1 - mean2) / max_mean)
    return alfa


def morph_dilatation(img):
    """
    A função tira o ruido e preenche o meio da a imagem
    :param img: img
    :return: Uma imagem
    """
    # Rect Kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_close)
    img = cv.GaussianBlur(img, (9, 9), 0)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img = cv.dilate(img, kernel_dilate, iterations=1)
    return img


def filtro_cinza(img):
    linhas , colunas , tipo = img.shape
    binary_mask = np.zeros((linhas, colunas))
    intervalo = 17
    for i in range(0, linhas):
        for j in range(0, colunas):
            valor = img[i, j, 0]
            if valor - intervalo < img[i, j, 1] < valor + intervalo:
                if valor - intervalo < img[i, j, 2] < valor + intervalo:
                    binary_mask[i, j] = 1
    imgprint("bus", binary_mask)
    return binary_mask


if __name__ == "__main__":
    img = cv.imread("estrada.png")

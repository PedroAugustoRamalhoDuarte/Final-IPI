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


def preencher_interior(img):
    h, w = img.shape
    seed = (w / 2, h / 2)

    mask = np.zeros((h + 2, w + 2), np.uint8)

    floodflags = 4
    floodflags |= cv.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    num, im, mask, rect = cv.floodFill(img, mask, seed, (255, 0, 0), (10,) * 3, (10,) * 3, floodflags)

    cv.imwrite("seagull_flood.png", mask)
    return mask


def morph_dilatation(img):
    # Rect Kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))
    kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))
    # imgprint(img)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_close)
    # img = cv.GaussianBlur(img, (9, 9), 0)
    # imgprint(img)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # imgprint(img)
    # img = cv.dilate(img, kernel_dilate, iterations=1)
    return img


if __name__ == "__main__":
    img = cv.imread("edge.png", cv.CAP_MODE_GRAY)
    img = morph_dilatation(img)

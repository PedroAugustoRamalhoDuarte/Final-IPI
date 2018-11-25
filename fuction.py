import cv2 as cv


def imgprint(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)


def morph_dilatation(img):
    # Rect Kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
    # imgprint(img)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_close)
    # imgprint(img)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # imgprint(img)
    # img = cv.dilate(img, kernel, iterations=2)
    # imgprint("final", img)
    return img


if __name__ == "__main__":
    img = cv.imread("edge.png", cv.CAP_MODE_GRAY)
    img = morph_dilatation(img)
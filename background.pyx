import cython

@cython.boundscheck(False)
cpdef unsigned char[:, :] self_background(unsigned char [:, :] image, unsigned char [:, :] fore, unsigned char [:, :] back):
    cdef int i, j, lin, col
    lin = image.shape[0]
    col = image.shape[1]

    for i in range(0, lin):
        for j in range(0, col):
            if image[i, j] > 0:
                image[i, j] = fore[i, j]
            else:
                image[i, j] = back[i, j]
    return image


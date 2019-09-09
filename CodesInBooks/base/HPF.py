import cv2
import numpy as np
from scipy import ndimage

kernel_3x3=np.array([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]])
#
# kernel_3x3=np.array([[[-1, -1, -1],
#                      [-1, 8, -1],
#                      [-1, -1, -1]],[[-1, -1, -1],
#                      [-1, 8, -1],
#                      [-1, -1, -1]],[[-1, -1, -1],
#                      [-1, 8, -1],
#                      [-1, -1, -1]]])

kernel_5x5=np.array([[-1, -1, -1, -1, -1],
                     [-1, 1, 2, 1, -1],
                     [-1, 2, 4, 2, -1],
                     [-1, 1, 2, 1, -1],
                     [-1, -1, -1, -1, -1]])
# read image in BGR channel
# img = cv2.imread("../data/imgs/l_hires.jpg")
# read image in gray scale
img = cv2.imread("../data/imgs/l_hires.jpg", 0)

# k3 = ndimage.convolve(img, kernel_3x3 / 3)
k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)
blurred = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - blurred
cv2.imshow("3x3", k3)
cv2.imshow("5x5", k5)
cv2.imshow("g_hpf", g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()
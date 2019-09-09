import numpy as np
import cv2
import pandas as pd
import warnings
import os

# pd.set_option("max_columns",1000)
# pd.set_option("max_row",300)
# pd.set_option("display.float_format", lambda x: '%.3f' % x)
# warnings.filterwarnings('ignore')

def show(img):
    cv2.imshow('moli', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = np.zeros((5,4), dtype=np.uint8)

img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# RGB与HSV的联系
# 从上面的直观的理解，把RGB三维坐标的中轴线立起来，并扁化，就能形成HSV的锥形模型了。
# 但V与强度无直接关系，因为它只选取了RGB的一个最大分量。而RGB则能反映光照强度（或灰度）的变化。
# v = max(r, g, b)
moli = cv2.imread("imgs/moli.jpg")

cv2.imwrite('imgs/moli.png', img = moli)
moli = cv2.imread("imgs/moli.jpg", cv2.IMREAD_GRAYSCALE)
h,w = moli.shape

cv2.imwrite('imgs/moli_gray.png', img = moli)

moli_bytes = bytearray(moli)

moli = cv2.imread("imgs/moli.jpg")
h,w,u = moli.shape
moli_bytes = bytearray(moli)
toBGRImg = np.array(moli_bytes).reshape(h,w,3)
moli_bytes = bytearray(cv2.cvtColor(moli, cv2.COLOR_BGR2GRAY))
toGrayImg = np.array(moli_bytes).reshape(h,w)

# make an array of 120000 random bytes
randomByteArray = bytearray(os.urandom(120000))
flatNpArray = np.array(randomByteArray)
# Convert the array to make a 400*300 grayscale image
grayImage = flatNpArray.reshape(300,400)
# another method for random array that is more high efficiency
# np.random.randint(0,256,120000).reshape(300,400)
cv2.imwrite('imgs/random.png', grayImage)
# Covert the array to make a 400*100 color image
bgrImage = flatNpArray.reshape(100, 400, 3)
cv2.imwrite('imgs/random_color.png', bgrImage)

show(grayImage)

img = cv2.imread('imgs/moli.jpg')
show(img)

# change one item
# img[0,0] = [255,255,255]
print(img.item(150,120,0))

# make given channel(bgr) 255
img.itemset((150,120,0), 255)
print(img.item(150,120,0))
show(img)

# it's inefficient to change item useing circye, Using row indexing sovle it
img[:,:, 0] = 0
show(img)

print(img.item(150,120,1))

img = cv2.imread('imgs/moli.jpg')

roi_1 = img[0:200, 0:200]
img[200:400, 200:400] = roi_1

show(img)

img.size, img.shape, img.dtype


videoCapture = cv2.VideoCapture('imgs/moli.mp4')

fps = videoCapture.get(cv2.CAP_PROP_FPS)
fps

size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
size

videoWriter = cv2.VideoWriter('imgs/out/moli.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
videoWriter

success, frame = videoCapture.read()
while success:
    videoWriter.write(frame)
    success, frame = videoCapture.read()

cameraCapture = cv2.VideoCapture(0)
fps = 30 # an assumption
size =  (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('imgs/out/my_camera.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1
while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
cameraCapture.release()

# synchronize a set of cameras or a multihead camera
cameraCapture0 = cv2.VideoCapture(0)
cameraCapture1 = cv2.VideoCapture(0)
fps = 30 # an assumption
size =  (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('imgs/out/my_camera2.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
success0 = cameraCapture0.grab()
success1 = cameraCapture1.grab()
while success0 and success1:
    frame0 = cameraCapture0.retrieve()
    frame1 = cameraCapture1.retrieve()
    videoWriter.write(frame0)
    videoWriter.write(frame1)
cameraCapture0.release()
cameraCapture1.release()


clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCpture = cv2.VideoCapture(0)
windowName = 'MyWindow'
cv2.namedWindow(windowName)
cv2.setMouseCallback(windowName, onMouse)

print('Show camera feed, Click window or press any key to stop')
success, frame = cameraCpture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow(windowName, frame)
    success, frame = cameraCpture.read()
cv2.destroyWindow(windowName)
cameraCpture.release()
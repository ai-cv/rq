import cv2
import numpy as np
import imutils


img = cv2.pyrDown(cv2.imread("../data/imgs/hg.png", cv2.IMREAD_UNCHANGED))
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 0, 112, cv2.THRESH_BINARY)
# contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow("", contours)
# cv2.waitKey()
# cv2.destroyAllWindows()
# contours = contours[1] if imutils.is_cv3() else contours[0]
for c in image:
    # find bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # normalize coordinates to integers
    box = np.int0(box)
    # draw contours
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
    cv2.drawContours(img, c, 0, (0, 255, 0), 3)

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    img = cv2.circle(img, center, radius, (0, 0, 255), 2)

# cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
img = cv2.resize(img, (420, 580))
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()

#!/usr/bin/python3

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os



def main():

    for file in os.listdir("images"):
        if file.endswith("zoom14.jpg"):
            filepath = os.path.join("images", file)


            out = process_image(filepath)
            cv.imwrite(os.path.join("out", file), out)

def process_image(file):

    HSV_GREEN=69
    GREEN_RANGE=9
    NO_SAT = 40

    img_in = cv.imread(file)

    img_hsv = cv.cvtColor(img_in, cv.COLOR_BGR2HSV)

    img_col, img_sat, _ = cv.split(img_hsv)

    # filter input image
    img_col = cv.blur(img_col, (3, 3))
    img_col = cv.fastNlMeansDenoising(img_col, None, 3, 7, 21)

    img_sat = cv.blur(img_sat, (7,7))
    #img_sat = cv.fastNlMeansDenoising(img_sat, None, 5, 7, 35)

    # hist = cv.calcHist([img_col], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.show()


    # extract the green part from the color part and create a mask
    _, img_mask1 = cv.threshold(img_col, HSV_GREEN - GREEN_RANGE, 255, cv.THRESH_BINARY)
    _, img_mask2 = cv.threshold(img_col, HSV_GREEN + GREEN_RANGE, 255, cv.THRESH_BINARY_INV)
    img_mask = cv.bitwise_and(img_mask1, img_mask2)
    #cv.imwrite('mask_col.jpg', img_mask)

    # remove the parts which have allmost no color (low satturation)
    _, img_mask_nc = cv.threshold(img_sat, NO_SAT, 255, cv.THRESH_BINARY)
    #cv.imwrite('mask_sat.jpg', img_mask_nc)

    img_mask = cv.bitwise_and(img_mask, img_mask_nc)
    #cv.imwrite('mask.jpg', img_mask)


    img_mask =cv.erode(img_mask,(10,10))


    low_freq_mask = cv.medianBlur(img_mask,21)
    #cv.imwrite('mask_lf.jpg', low_freq_mask)

    o = getChildOrientation(low_freq_mask)


    img_in[img_mask>1] = (20,40,255)

    img_out = img_in[o[0]:o[1], o[2]:o[3]]

    return img_out

    #cv.imwrite('out6.jpg', img_out)



def getChildOrientation(image):
    #img, contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    img_inv = cv.bitwise_not(image)
    num_labels, labels, stats , centroids = cv.connectedComponentsWithStats(img_inv, 4, cv.CV_32S)

    print(num_labels)
    print(labels)
    print(stats)
    print(centroids)

    if num_labels != 3:
        print('unable to process')

    #image size
    img_height, img_width = image.shape

    # get the leftmost object and set it to zero
    for i in range(num_labels):
        cc = stats[i]
        if cc[cv.CC_STAT_LEFT] == 0:
            img_inv[labels==i]=0

        if cc[cv.CC_STAT_LEFT] + cc[cv.CC_STAT_WIDTH] == img_width and cc[cv.CC_STAT_HEIGHT] != img_height:
            print('child detected!')

            x1=cc[cv.CC_STAT_LEFT]
            y1=cc[cv.CC_STAT_TOP]
            x2=x1+cc[cv.CC_STAT_WIDTH]
            y2=y1+cc[cv.CC_STAT_HEIGHT]

            return [y1,y2,x1,x2]

    return [0, img_height, 0, img_width]


    #img_cont = cv.drawContours(im, contours , -1, (0, 255, 0), 3)
    #cv.imwrite('cont.jpg', img_inv)

if __name__ == "__main__":
    main()
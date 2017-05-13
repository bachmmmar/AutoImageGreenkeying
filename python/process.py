#!/usr/bin/python3

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import pre_processing_result as PreProResult
import border_result as BorderResult




def main():
    do_testimage=False

    input_directory="images"
    input_filter="zoom14.jpg"
    tmp_directory = "tmp"
    output_directory="out"
    background_image="background.jpg"

    result = PreProResult.PreProcessingResult()

    if not do_testimage:
        for in_file in os.listdir(input_directory):
            if in_file.endswith(input_filter):
                in_filepath = os.path.join(input_directory, in_file)
                out_filepath = os.path.join(tmp_directory, "{}.tiff".format(in_file))

                p=PreProcessor(result)
                p.process_file(in_filepath,out_filepath)

        result.print()

        post = PostProcessor(result,background_image)

        for r in result._result:
            out_filepath = os.path.join(output_directory, r.filename_in)
            post.addBackground(r, out_filepath)


    else:
        p = PreProcessor(result)
#        p.process_file('source.jpg', 'out.jpg')
        p.process_file('images/20170429T070432_zoom14.jpg', 'tmp.tiff')
        p.write_debug_images('dbg')

        result.print()

        post = PostProcessor(result, background_image)
        post.addBackground(result._result[0],"out.jpg")


class PostProcessor:
    def __init__(self, result, background):
        self._result = result

        self._img_size = self.getTargetImageSize()

        self._bg_img = cv.imread(background)
        h, w = self._bg_img.shape[:2]
        self._bg_size = (w,h)
        print('Background image size is {}x{}Pixels'.format(w, h))

        self._target_rescaleing = self.getTargetRescaling()

    def getTargetImageSize(self):
        height = []
        width = []

        for r in self._result._result:
            _, ycom = r.center_of_mass
            (w, h) = r.border.get_size()

            height.append(max(ycom,h/2)*2)
            width.append(w)

        max_height = int(max(height))
        max_width = int(max(width))
        min_width = int(min(width))

        print('Target maximum image size would be {}x{}Pixels'.format(max_width,max_height))

        return (min_width,max_height)

    def getTargetRescaling(self):
        scale_w = self._bg_size[0] / self._img_size[0]
        scale_h = self._bg_size[1] / self._img_size[1]
        scale = round(min(scale_h, scale_w),3)
        print('Target rescaling by {}x'.format(scale))
        print('New Target image size will be {}x{}Pixels'.format(int(self._img_size[0]*scale), int(self._img_size[1]*scale)))
        return scale

    def addBackground(self, result, output_file):
        img_in = cv.imread(result.filename_out, cv.IMREAD_UNCHANGED)
        img_scaled = cv.resize(img_in,None, fx=self._target_rescaleing, fy=self._target_rescaleing)

        bg = copy.deepcopy(self._bg_img)

        img_rgba = np.zeros((self._bg_size[1], self._bg_size[0],4), np.uint8)

        _, ycom = result.center_of_mass
        (bg_width, bg_height)=self._bg_size
        fg_height, fg_width = img_scaled.shape[:2]
        y=int(bg_height/2-(ycom*self._target_rescaleing))
        yh=min(bg_height, y+fg_height)
        width = min(bg_width, fg_width)

        print('Place image to {}:{}'.format(y,yh))
        img_rgba[y:yh, 0:width] = img_scaled[0:yh-y,0:width]

        b, g, r, a = cv.split(img_rgba)
        img_rgb = cv.merge((b, g, r))

        bg[a>1]=img_rgb[a>1]

        print('Write outputfile {}'.format(output_file))
        cv.imwrite(output_file, bg)

        #dest_h = 2700
        # dest_w = 3200
        # dest_img = np.zeros((dest_h,dest_w,3), np.uint8)
        # dest_img[:,:] = (0,0,255)      # (B, G, R)
        #
        # x1, x2, y1, y2 = res.border.get_ranges()
        # w,h = res.border.get_size()
        # _, ycom = res.center_of_mass
        # y=int(dest_h/2-ycom)
        # x=20
        # dest_img[y:y+h, x:x+w] = img_in[y1:y2, x1:x2]
        #
        # return dest_img



# find peak and width
class PreProcessor:

    def __init__(self, result):
        self.debug_images = []
        self._result = result
        self._in_file = ''
        self._out_file = ''

    class DebugImage:
        def __init__(self, name, image):
            self._name = name
            self._image = image

    class NoChildFound(RuntimeError):
        def __init__(self, arg):
            self.args = arg

    def process_file(self, in_file, out_file):
        self._in_file = in_file
        self._out_file = out_file

        img_in = cv.imread(in_file)
        try:
            out = self.process_image(img_in)
            cv.imwrite(out_file, out)
        except self.NoChildFound as e:
            print(e)
            self.write_debug_images(out_file)


    def add_debug_image(self, name, image):
        self.debug_images.append(self.DebugImage(name,image))

    def write_debug_images(self, path):
        out_dir = '{}.d'.format(path)
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass
        for img in self.debug_images:
            cv.imwrite(os.path.join(out_dir, img._name), img._image)

    def process_image(self, img_in):

        HSV_GREEN=69
        GREEN_RANGE=13
        NO_SAT = 40

        img_hsv = cv.cvtColor(img_in, cv.COLOR_BGR2HSV)

        img_col, img_sat, _ = cv.split(img_hsv)

        # filter input image
        img_col = cv.blur(img_col, (3, 3))
        img_col = cv.fastNlMeansDenoising(img_col, None, 3, 7, 21)

        img_sat = cv.blur(img_sat, (7,7))
        #img_sat = cv.fastNlMeansDenoising(img_sat, None, 5, 7, 35)

        # get peak green value, assuming green ist the most used collor
        hist = np.array(cv.calcHist([img_col], [0], None, [256], [0, 256]))
        (k, _) = np.where(hist == hist.max())
        HSV_GREEN = k[0]
        print('HSV peak green @ {}'.format(HSV_GREEN))
        #print(x)
        # plt.plot(hist)
        # plt.xlim([0, 256])
        # plt.show()


        # extract the green part from the color part and create a mask
        _, img_mask1 = cv.threshold(img_col, HSV_GREEN - GREEN_RANGE, 255, cv.THRESH_BINARY)
        _, img_mask2 = cv.threshold(img_col, HSV_GREEN + GREEN_RANGE, 255, cv.THRESH_BINARY_INV)
        img_mask_col = cv.bitwise_and(img_mask1, img_mask2)
        self.add_debug_image('mask_col.jpg', img_mask_col)

        # extract the parts which have allmost no color (low satturation)
        _, img_mask_nc = cv.threshold(img_sat, NO_SAT, 255, cv.THRESH_BINARY)
        self.add_debug_image('mask_sat.jpg', img_mask_nc)

        # remove the parts which have allmost no color from color mask
        img_mask_comb = cv.bitwise_and(img_mask_col, img_mask_nc)
        self.add_debug_image('mask_comb.jpg', img_mask_comb)

        # remove noise from mask
        img_mask = cv.erode(img_mask_comb,(10,10))
        self.add_debug_image('mask.jpg', img_mask)


        # crete mask for orientation determination
        low_freq_mask = cv.medianBlur(img_mask,21)
        self.add_debug_image('mask_lf.jpg', low_freq_mask)

        res = self.get_child_orientation(low_freq_mask)

        # create alpha channel image
        b_channel, g_channel, r_channel = cv.split(img_in)
        img_mask_not = cv.bitwise_not(img_mask)
        img_RGBA = cv.merge((b_channel, g_channel, r_channel, img_mask_not))

        #return the croped image
        x1, x2, y1, y2 = res.border.get_ranges()
        return img_RGBA[y1:y2, x1:x2]

        # # set pixels to red
        # img_in[img_mask>1] = (20,40,255)
        #
        # dest_h = 2700
        # dest_w = 3200
        # dest_img = np.zeros((dest_h,dest_w,3), np.uint8)
        # dest_img[:,:] = (0,0,255)      # (B, G, R)
        #
        # x1, x2, y1, y2 = res.border.get_ranges()
        # w,h = res.border.get_size()
        # _, ycom = res.center_of_mass
        # y=int(dest_h/2-ycom)
        # x=20
        # dest_img[y:y+h, x:x+w] = img_in[y1:y2, x1:x2]
        #
        # return dest_img

    def get_child_orientation(self, image):
        CHILD_CENTER = (1232, 2259) #(2259, 1232)

        # segment image into components
        img_inv = cv.bitwise_not(image)
        num_labels, labels, stats , centroids = cv.connectedComponentsWithStats(img_inv, 4, cv.CV_32S)

        # get the label of the expected position
        label = labels[CHILD_CENTER]

        if num_labels > 10:
            raise self.NoChildFound('Too many labels detected ({})'.format(num_labels))
            print(num_labels)
            print(labels)
            print(stats)
            print(centroids)

        print('Child label: {}'.format(label))
        cc = stats[label]

        border = BorderResult.BorderResult(cc[cv.CC_STAT_LEFT], cc[cv.CC_STAT_TOP], cc[cv.CC_STAT_WIDTH], cc[cv.CC_STAT_HEIGHT])

        #calculate centroids refering to border
        x = centroids[label,0]
        y = centroids[label,1]
        x = int(x - cc[cv.CC_STAT_LEFT])
        y = int(y - cc[cv.CC_STAT_TOP])

        res = PreProResult.Result(self._in_file, self._out_file, (x,y), border, 0)
        self._result.add_result(res)

        return res

if __name__ == "__main__":
    main()

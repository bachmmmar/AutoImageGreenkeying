import cv2 as cv
import numpy as np
import copy

class PostProcessor:
    def __init__(self, result, background, use_min_not_max:bool):
        self._result = result
        self._use_min_not_max = use_min_not_max;

        self._img_size = self.getTargetImageSize()
        print('Optimum Target image size would be {}x{}Pixels'.format(self._img_size[0], self._img_size[1]))

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

        if self._use_min_not_max:
            return (min_width,max_height)
        else:
            return (max_width, max_height)

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

        if self._use_min_not_max:
            fg_width = int(self._img_size[0]*self._target_rescaleing)

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


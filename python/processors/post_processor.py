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

    def __init__(self):
        self._result = None
        self._use_min_not_max = False
        self._img_size = (12,24)
        self._bg_img = None
        self._target_rescaleing = 1

    def getTargetImageSize(self):
        height = []
        width = []

        for r in self._result._result:
            _, ycom = r.center_of_mass
            (w, h) = r.border.get_size()
            ang = r.rotation

            (w,h) = self.calculate_rotated_dimension(w,h,ang)

            height.append(max(ycom,h/2)*2)
            width.append(w)

        max_height = int(max(height))
        max_width = int(max(width))
        min_height = int(min(height))

        if self._use_min_not_max:
            return (max_width,min_height)
        else:
            return (max_width, max_height)

    def calculate_rotated_dimension(self, w,h,ang_deg):
        ang = ang_deg/180*np.pi
        hn = np.ceil(abs(w * np.sin(ang)) + abs(h * np.cos(ang)))
        wn = np.ceil(abs(w * np.cos(ang)) + abs(h * np.sin(ang)))
        return (wn, hn)

    def getTargetRescaling(self):##
        scale_w = self._bg_size[0] / self._img_size[0]
        scale_h = self._bg_size[1] / self._img_size[1]
        scale = round(min(scale_h, scale_w),3)
        print('Target rescaling by {}x'.format(scale))
        print('New Target image size will be {}x{}Pixels'.format(int(self._img_size[0]*scale), int(self._img_size[1]*scale)))
        return scale

    def find_img_top(self, mask):
        in_h, in_w = mask.shape[:2]

        for l in range(0,in_h-1):
            line = mask[l:l+1, 0:in_w]
            if np.max(line) > 0:
                return l

        return 0

    def addBackground(self, result, output_file):
        img_in = cv.imread(result.filename_out, cv.IMREAD_UNCHANGED)
        in_h, in_w = img_in.shape[:2]

        (rot_w, rot_h) = self.calculate_rotated_dimension(in_w, in_h, result.rotation)

        # do the image rotation and shift
        end_w = int(rot_w*self._target_rescaleing)
        end_h = int(rot_h*self._target_rescaleing)
        rescaledsize = (end_w, end_h)
        rotationcenter = (int(in_w / 2), int(in_h / 2))
        M = cv.getRotationMatrix2D(rotationcenter, -result.rotation, self._target_rescaleing)
        # set image shift
        if result.rotation >0 :
            M[0][2] = np.ceil((rot_w-in_w)*self._target_rescaleing)
            M[1][2] = 0
        else:
            M[0][2] = 0 #np.ceil((rot_w-in_w)*self._target_rescaleing)
            M[1][2] = np.ceil((rot_h-in_h)*self._target_rescaleing)

        img_rot = cv.warpAffine(img_in, M, rescaledsize, borderMode=cv.BORDER_CONSTANT, borderValue=(0,255,0,0))

        # find first lines where the head starts and remove it
        _, _, _, a = cv.split(img_rot)
        t = self.find_img_top(a)
        img_scaled = img_rot[t:end_h, 0:end_w]
        fg_height, fg_width = img_scaled.shape[:2]

        # center of mass correction
        xcom, _ = result.center_of_mass
        xcom = fg_width/in_w*xcom

        # take the background and make a copy
        bg = copy.deepcopy(self._bg_img)
        (bg_width, bg_height) = self._bg_size

        # select the height depending on parameters
        if self._use_min_not_max:
            h = int(self._img_size[0]*self._target_rescaleing)
            if h>fg_height:
                print("wrong calculated min height {} vs. {}".format(h,fg_height))
            else:
                fg_height = h

        x=int(bg_width/2-xcom)
        xh=min(bg_width, x+fg_width)
        height = min(bg_height, fg_height)

        # create image with background size and place foreground into it
        print('Place image to {}:{}'.format(x,xh))
        img_rgba = np.zeros((bg_height, bg_width, 4), np.uint8)
        img_rgba[0:height, x:xh] = img_scaled[0:height,0:xh-x]

        # take only rgb part
        b, g, r, a = cv.split(img_rgba)
        img_rgb = cv.merge((b, g, r))

        # place the rgb part into background where alpha channel is set to intransparent
        bg[a>1]=img_rgb[a>1]

        print('Write outputfile {}'.format(output_file))
        cv.imwrite(output_file, bg)


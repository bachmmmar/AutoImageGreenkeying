import os
import cv2 as cv
import numpy as np

from .result import *


# find peak and width
class PreProcessor:

    def __init__(self, result, source_dir, output_dir):
        self.debug_images = []
        self._result = result
        self._source_dir = source_dir
        self._output_dir = output_dir
        self._in_file = ''
        self._in_filepath = ''
        self._rotcw = False
        self._face_cascade = cv.CascadeClassifier('cascade_clasifier/haarcascade_frontalface_default.xml')
        self._eye_cascade = cv.CascadeClassifier('cascade_clasifier/haarcascade_eye.xml')

    class DebugImage:
        def __init__(self, name, image):
            self._name = name
            self._image = image

    class NoChildFound(RuntimeError):
        def __init__(self, arg):
            self.args = arg

    def process_file(self, in_file, rotcw = False, main_object_loc = (2259, 1232)):
        self._in_file = in_file
        self._in_filepath = os.path.join(self._source_dir, in_file)
        self._out_filepath = os.path.join(self._output_dir, "{}.tiff".format(in_file))
        self._rotcw = rotcw
        self._main_object_loc = main_object_loc

        print(self._in_filepath)
        img_in = cv.imread(self._in_filepath)

        if rotcw:
            img_in = cv.transpose(img_in)
            img_in = cv.flip(img_in, 1)

        try:
            out = self.process_image(img_in)
            cv.imwrite(self._out_filepath, out)
            #self.write_debug_images(self._out_filepath)
        except self.NoChildFound as e:
            print(e)
            self.write_debug_images(self._out_filepath)


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

        GREEN_RANGE=14
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
        #img_mask_noise = cv.dilate(img_mask_comb,np.ones((2,2),np.uint8))
        #self.add_debug_image('mask_noise.jpg', img_mask_noise)


        # crete mask for orientation determination
        low_freq_mask = cv.medianBlur(img_mask_comb,21)
        self.add_debug_image('mask_lf.jpg', low_freq_mask)

        res, selected_comp = self.get_child_orientation(low_freq_mask)

        # invert component and make it smaller
        img_comp = cv.bitwise_not(selected_comp)
        img_comp_small = cv.dilate(img_comp, np.ones((10,10),np.uint8))
        self.add_debug_image('mask_selobj.jpg', img_comp_small)

        # combine mask to ensure no holes in object (details at border will be kept by other masks)
        img_mask_final = cv.bitwise_and(img_comp_small, img_mask_comb)
        self.add_debug_image('mask.jpg', img_mask_final)

        # create alpha channel image
        b_channel, g_channel, r_channel = cv.split(img_in)
        img_mask_not = cv.bitwise_not(img_mask_final)
        img_RGBA = cv.merge((b_channel, g_channel, r_channel, img_mask_not))

        #return the croped image
        x1, x2, y1, y2 = res.border.get_ranges()
        croped_img = img_RGBA[y1:y2, x1:x2]

        res.face = self.face_recognition(croped_img)

        self._result.add_result(res)

        return croped_img

    def face_recognition(self, img_colr):
        min_face_size=(300,300)
        max_face_size=(1000,1000)
        min_eye_size = (100, 100)
        max_eye_size = (190, 190)

        gray = cv.cvtColor(img_colr, cv.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.05, 4, minSize=min_face_size, maxSize=max_face_size)

        img_dbg = img_colr.copy()

        face = FaceResult()

        for (x, y, w, h) in faces:
            cv.rectangle(img_dbg, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img_dbg[y:y + h, x:x + w]
            eyes = self._eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=min_eye_size, maxSize=max_eye_size)
            if len(eyes) == 2:
                print('found {0} face(s) with two eyes'.format(len(faces)))
                (elx, ely, elw, elh) = eyes[0]
                (erx, ery, erw, erh) = eyes[1]
                face = FaceResult( (int(x+w/2), int(y+h/2)), \
                                   (int(elx+elw/2), int(ely+elh/2)), \
                                   (int(erx+erw/2), int(ery+erh/2)))

            print('Face at {0}x{1} with {2}x{3}'.format(y, x, h, w))
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                print('eye at {0}x{1} with {2}x{3}'.format(y+ey,x+ex,eh, ew))

        self.add_debug_image('face.jpg', img_dbg)
        return face

    def determine_rotation(self, selected_comp):
        # get all points from specified label and fit them
        k = np.transpose(np.where(np.equal(selected_comp, 255)))
        re = cv.fitLine(k, cv.DIST_L12, 0, 0.1, 0.1)
        ang = np.arctan((re[1]) / (re[0])) * 180 / np.pi

        if not self._rotcw:
            if ang < 0:
                ang += 90
            else:
                ang -= 90
        return ang[0]

    def get_child_orientation(self, image):

        # segment image into components
        img_inv = cv.bitwise_not(image)
        num_labels, labels, stats , centroids = cv.connectedComponentsWithStats(img_inv, 4, cv.CV_32S)

        # get the label of the expected position
        label = labels[self._main_object_loc]

        if num_labels > 20:
            raise self.NoChildFound('Too many labels detected ({})'.format(num_labels))
            print(num_labels)
            print(labels)
            print(stats)
            print(centroids)

        sizeh, sizew = labels.shape[:2]
        selected_comp = np.zeros((sizeh, sizew, 1), np.uint8)
        selected_comp[label==labels] = 255

        cc = stats[label]

        border = BorderResult(cc[cv.CC_STAT_LEFT], cc[cv.CC_STAT_TOP], cc[cv.CC_STAT_WIDTH], cc[cv.CC_STAT_HEIGHT])

        ang = self.determine_rotation(selected_comp)

        #calculate centroids refering to border
        x = centroids[label,0]
        y = centroids[label,1]
        x = (x - border._left)
        y = (y - border._top)

        res = PreProcessingResult(self._in_file, self._out_filepath, (int(x),int(y)), border, ang)

        return res, selected_comp
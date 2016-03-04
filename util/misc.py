"""
Created on Nov 19, 2015

@author: agp

reverted back to an earlier version
TODO refactor
"""
import os
import subprocess
import random
import string

from SimpleCV import Image, cv2, FeatureSet

import logging
import numpy as np
from math import sqrt
# import matplotlib.pyplot as plt
from cv2 import cv


class CumulativeSum:
    def __init__(self, cumulative_sum=0):
        self.sum = cumulative_sum

    def __call__(self, a):
        self.sum += a
        return self.sum


def gradient_1d(function_points):
    if len(function_points) < 2:
        return function_points
    gradient = []
    for i, _ in enumerate(function_points):
        left = function_points[max(i - 1, 0)]
        right = function_points[min(i + 1, len(function_points) - 1)]
        gradient.append((right - left) / 2.0)  # + (right - left) / 2.0) / 3.0)
    return gradient


def get_horizontal_params(image2, x, y, width, height):
    hor = []
    hist = []
    for i in range(y, min(y + height, image2.height)):
        ctr = 0.0
        h_ctr = image2[x, i][0]
        for j in range(x, min(x + width, image2.width)):
            diff = abs(image2[j, i][0] - image2[max(j - 1, 0), i][0])
            if diff > 0.2:
                ctr += diff
            else:
                ctr += abs(image2[j, i][0] - image2[j, max(i - 1, 0)][0])
            h_ctr += image2[j, i][0]  # image
        hist.append(ctr)
        hor.append(h_ctr)
    return hist, hor


def get_vertical_params(image, x, y, width, height):
    params = []
    thickness_sum = 0.0
    intensity_sum = 0.0
    char_width = 0.0
    n_chars = 1.0
    prev_cut = 0
    in_char = False
    for i in range(x, x + width):
        tops = [-1, -1, -1, ]
        bottoms = [-2, -2, -2, ]
        left_right = 0.0
        up_down = 0.0
        intensity = 0.0
        for j in range(y, y + height):
            p = image[i, j][0]
            left_right += abs(p - image[max(i - 1, 0), j][0])
            up_down += abs(p - image[i, max(j - 1, 0)][0])
            intensity += p
            if p < 255:
                if tops[0] == -1:
                    tops[0] = j
                bottoms[0] = j
                if p < 165:
                    if tops[1] == -1:
                        tops[1] = j
                    bottoms[1] = j
                    if p < 75:
                        if tops[2] == -1:
                            tops[2] = j
                        bottoms[2] = j
        changes = [left_right, up_down]
        thickness = (sum(bottoms) / 3.0 - sum(tops) / 3.0) + 1
        thickness_sum += thickness
        intensity_sum += intensity
        if thickness == 0:
            if in_char:
                char_width += i - prev_cut
                n_chars += 1
                in_char = False
            prev_cut = i
        else:
            in_char = True
        params.append((intensity, changes, bottoms, tops, thickness))
    avg_thickness = thickness_sum / width
    avg_intensity = intensity_sum / width
    char_width /= n_chars
    return char_width, avg_thickness, avg_intensity, params


def frame_image(image, color=(0, 0, 0), min_width=8):
    image.drawRectangle(0, 0, image.width - 3, image.height - 3,
                        width=min(min_width, 1 + min(image.width, image.height) / 20),
                        color=color)
    return image.applyLayers()


def adaptive_histeq(img, img_name=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if isinstance(img, basestring):
        img = cv2.imread(img, 0)
    else:  # if hasattr(img,'getGrayNumpyCv2'):
        img = img.getGrayNumpyCv2()
    cl1 = clahe.apply(img)
    if img_name is not None:
        cv2.imwrite(img_name, cl1)
    if Image(cl1) is None:
        logging.warning("Image for the adaptive histeq does not exist")
        return None
    return Image(cl1).transpose()


def find_contours(image):
    mMemStorage = cv.CreateMemStorage()
    contours = cv.FindContours(image._getGrayscaleBitmap(),
                               mMemStorage,
                               cv.CV_RETR_CCOMP,
                               cv.CV_CHAIN_APPROX_SIMPLE)
    return contours


def contours_to_boxes(contours, rectangles, max_width, max_height, x_pad=0, y_pad=0,
                      filter=lambda bb: bb[2] < 2 or bb[3] < 6):
    while contours is not None and len(contours) != 0:
        bb = cv.BoundingRect(contours)
        contours = contours.h_next()
        if filter(bb):
            logging.warning("contour is filtered")
            continue
        else:
            box = [max(x_pad + bb[0] - 2, 0), max(y_pad + bb[1] - 2, 0), min(max_width, bb[2] + 2),
                   min(max_height, bb[3] + 2)]
            rectangles.append(box)
    return rectangles


def find_candidate_text_components(swt, input_image=None, level=0):
    lines = swt  # morph_swt(swt)
    lines.show()
    # raw_input("okidoki")
    if input_image is None:
        input_image = lines.invert()
    mask = filter_contours(swt.invert(), input_image, invert=True, min_size=5, level=level)
    lines = mask.invert() + ((mask.invert() + input_image.invert()).invert()).invert()
    # time.sleep(5)
    return find_contours(lines.invert())


def filter_components(swt, input_image, candidate_boxes):
    text_boxes = []
    '''TODO filter boxes with centroids inside other boxes'''
    return candidate_boxes


def find_text_components(swt, input_image):
    if isinstance(swt, basestring):
        swt = Image(swt)
    if isinstance(input_image, basestring):
        input_image = Image(input_image)
    # img = morph_swt(swt)
    # img.show()
    # swt.show()
    # raw_input("dodo")
    contours = find_candidate_text_components(swt, swt.morphClose().invert())
    candidate_boxes = []
    contours_to_boxes(contours, candidate_boxes, swt.width, swt.height)
    # decide whether the rectangle contains text
    return filter_components(swt, input_image, candidate_boxes)


"""def find_candidate_text_components(swt):
    kernel = np.ones((15, 3), np.uint8)
    lines = morph_swt(swt.morphClose())  # Image(cv2.morphologyEx(swt.getNumpy(), cv2.MORPH_OPEN, kernel)).erode().dilate().erode(3)
    mask = filter_contours(lines.invert(), swt.invert(), invert=True, min_size=5)
    # mask.show()
    lines = mask.invert() + ((mask.invert() + swt).invert()).invert()
    # lines.show()
    return find_contours(lines.invert())


def filter_components(swt, input_image, candidate_boxes):
    text_boxes = []
    '''TODO filter boxes with centroids inside other boxes'''
    return candidate_boxes

def find_text_components(swt, input_image):
    if isinstance(swt, basestring):
        swt = Image(swt)
    if isinstance(input_image, basestring):
        input_image = Image(input_image)
    contours = find_candidate_text_components(swt)
    candidate_boxes = []
    contours_to_boxes(contours, candidate_boxes, swt.width, swt.height)
    return filter_components(swt, input_image, candidate_boxes)"""


def get_components(input_image="image.jpg", dob=0):
    """returns a list of rectangles (pair of upper left and lower right points)"""
    assert (isinstance(input_image, basestring))
    if not os.path.isfile(input_image[:-4] + "_SWT.png"):
        subprocess.call(['data/external_SWT/./DetectText',
                         input_image, 'result.png', '%d' % dob])
        subprocess.call(['mv', 'SWT.png', "%s_SWT.png" % input_image[:-4]])

    return find_text_components(morph_swt(Image("%s_SWT.png" % input_image[:-4])).invert(), input_image)
    # with open("componentsWithBoxes.txt") as components:
    #    return [[point.strip().split(",") for point in line.strip().split(";")] for line in components]


def get_sample_letter_bb(img):
    """try to return a bounding box for a sample letter,
    if failed, return the full box bounding the image"""
    fs = img.findBlobs()
    if fs is not None:
        for f in fs:
            cropped = f.crop()
            colors = cropped.getGrayHistogramCounts(bins=5)
            edges = cv2.countNonZero(cropped.morphGradient().getGrayNumpyCv2())
            rec_perimeter = (cropped.width + cropped.height) * 2
            if edges > rec_perimeter * 2.5 and 0.6 < f.aspectRatio() < 3 and abs(
                            colors[0][1] - colors[1][1]) > 2:  # something that is like a letter
                hist = cropped.getNormalizedHueHistogram()
                return f.boundingBox()
    return 0, 0, img.width, img.height


def get_mean(val_list):
    return sum(val_list) / float(len(val_list) + 1)


def get_variance(val_list, mean):
    variance = sum([(a - mean) ** 2 for a in val_list]) / float(len(val_list) + 1)
    return variance ** 0.5


def filter_features_in_line(line):
    pass


def filter_features_in_word(word):
    word = sorted(word)
    if len(word) < 2:
        return word
    [x, y, w, h] = word[0].rectangle
    prev_color = word[0].feature.meanColor()
    for f in word[1:]:
        [x, y, w, h] = f.rectangle


def smart_filter(fs):
    rectangle_features = [RectangleElement(rectangle=f.boundingBox(), feature=f) for f in fs]
    previous = None
    previous_space = 100
    word = []
    sentence = []
    sentences = []
    for rectangle_f in sorted(rectangle_features):
        if previous is None:
            assert (len(word) == 0)
            word.append(rectangle_f.feature)
        else:
            if rectangle_f.rectangle[1] + rectangle_f.rectangle[3] / 2 > previous.rectangle[1] + \
                    previous.rectangle[3]:
                # end of a sentence/line, start a new sentence
                sentence.append(word)
                sentences += filter_features_in_line(sentence)
                sentence = []
                word = [rectangle_f.feature]
            elif rectangle_f.rectangle[0] - (
                        previous.rectangle[0] + previous.rectangle[2]) > 1 + previous_space * 1.2 + (
                        rectangle_f.rectangle[2] + previous.rectangle[2]) * 0.05:
                # end of a word, start a new word
                sentence += filter_features_in_word(word)
                word = [rectangle_f.feature]
            else:
                word.append(rectangle_f.feature)
            previous_space = max(rectangle_f.rectangle[0] - (previous.rectangle[0] + previous.rectangle[2]), 0)
        previous = rectangle_f
    sentence += filter_features_in_word(word)
    sentences += filter_features_in_line(sentence)
    return FeatureSet(sentences)


def filter_contours(image, input_image, invert=True, min_size=5, level=0):
    """TODO research on invariant features for filtering"""
    img = frame_image(image)
    # img.show()
    # raw_input("img")
    # input_image.show()
    # raw_input("image")
    area = image.width * image.height
    image_mask = Image((img.width, img.height))
    # logging.info(area)
    block_size = 2 * (int(area ** 0.7)) + 1
    # logging.info(block_size)

    pad = min(3, 1 + min(image.width, image.height) // 150)
    it = 0
    """myMask = Image((img.width, img.height))
    myMask = myMask.floodFill((0, 0), color=(128, 128, 128))
    mask = img.threshold(128)
    myMask = (myMask - mask.dilate(3) + mask.erode(3))
    result = img.watershed(mask=myMask, useMyMask=True)
    result.show()
    raw_input("h")"""
    fs = img.findBlobs()  # threshblocksize=11, threshconstant=1, minsize=min_size)
    if fs is not None:
        logging.info(len(fs))
        '''blob_areas = fs.area()
        avg_blob_area = get_mean(blob_areas)
        blob_area_std = get_variance(blob_areas, avg_blob_area)

        crops = [input_image.crop(*f.boundingBox()) for f in fs]
        edge_area_ratios = [float(crop.width * crop.height) /
                            (cv2.countNonZero(crop.morphGradient().getGrayNumpyCv2()) + 1) for crop in crops]
        avg_edge_area_ratio = get_mean(edge_area_ratios)
        edge_area_std = get_variance(edge_area_ratios, avg_edge_area_ratio)'''
        heights = fs.height()
        avg_height = get_mean(heights)
        height_std = get_variance(heights, avg_height)
        if len(fs) < 10:
            height_std = 2.5 * (height_std + 1)
        fs = fs.filter(fs.height() < avg_height + 6 * height_std)
        blob_areas = fs.area()
        avg_blob_area = get_mean(blob_areas)
        blob_area_std = get_variance(blob_areas, avg_blob_area)
        if len(fs) < 10:
            blob_area_std = 2.5 * (blob_area_std + 1)
        fs = fs.filter(fs.area() < avg_blob_area + 8 * blob_area_std)
        fs = fs.filter([(i and j) or (k and m) for i, j, k, m in
                        zip(fs.height() - fs.width() <= 0, fs.width() / fs.height() < 16, fs.height() - fs.width() > 0,
                            fs.height() / fs.width() < 8)])
        logging.info(len(fs))
        # fs = smart_filter(fs)
        crops = [input_image.crop(*f.boundingBox()) for f in fs]
        edge_area_ratios = [float(crop.width * crop.height) /
                            (cv2.countNonZero(crop.morphGradient().getGrayNumpyCv2()) + 1) for crop in crops]
        avg_edge_area_ratio = get_mean(edge_area_ratios)
        edge_area_std = get_variance(edge_area_ratios, avg_edge_area_ratio)

        blob_areas = fs.area()
        avg_blob_area = get_mean(blob_areas)
        blob_area_std = get_variance(blob_areas, avg_blob_area)
        heights = fs.height()
        avg_height = get_mean(heights)
        height_std = get_variance(heights, avg_height)
        widths = fs.width()
        avg_width = get_mean(widths)
        width_std = get_variance(heights, avg_width)
        rec_areas = [w * h for w, h in zip(widths, heights)]
        avg_rec_area = get_mean(rec_areas)
        rec_area_std = get_variance(rec_areas, avg_rec_area)
        aspect_ratios = fs.aspectRatios()
        avg_aspect_ratio = get_mean(aspect_ratios)
        aspect_ratio_std = get_variance(aspect_ratios, avg_aspect_ratio)
        if len(fs) < 10:
            edge_area_std = 2.5 * (edge_area_std + 1)
            blob_area_std = 2.5 * (blob_area_std + 1)
            aspect_ratio_std = 2.5 * (aspect_ratio_std + 1)
            rec_area_std = 2.5 * (rec_area_std + 1)
            width_std = 2.5 * (width_std + 1)
            height_std = 2.5 * (height_std + 1)

        for f in fs:
            [x, y, w, h] = f.boundingBox()
            box_area = w * h
            # img.drawRectangle(x, y, w, h, color=(0, 255, 255))
            # logging.info(f.aspectRatio(), f.area(), w * h)
            cropped = input_image.crop(x, y, w, h)
            blob_area = cv2.countNonZero(cropped.getGrayNumpyCv2())
            if blob_area < 10:
                # logging.info("too small")
                continue
            colors = cropped.getGrayHistogramCounts(bins=4)
            # logging.info(colors)
            # f.crop().show()

            edges = cv2.countNonZero(cropped.morphGradient().getGrayNumpyCv2())
            edge_area_ratio = float(blob_area) / (edges + 1)
            bin_edges = cv2.countNonZero(cropped.blur(3).binarize().morphGradient().getGrayNumpyCv2())
            rec_perimeter = (cropped.width + cropped.height)
            fs2 = cropped.findBlobs()
            weird_blob = 0
            if fs2 == None:
                img.drawRectangle(x, y, w, h, color=(127, 0, 0), width=-1)
                # logging.info("no blobs inside")
                continue
            for f2 in fs2:
                if f2.aspectRatio() > 15:
                    weird_blob += 1

            # cropped.show()
            # raw_input("n%d,%d,%d" % (edges, rec_perimeter, f.perimeter()))
            if level > 0 or not (abs(h - avg_height) > 3.5 * height_std or abs(w - avg_width) > 5.5 * width_std or
                                         weird_blob > len(fs2) / 3.25 or abs(w * h - avg_rec_area) > 4 * rec_area_std or
                                         cropped.width > img.width * 0.9 or cropped.height > img.height * 0.9 or
                                         edges < rec_perimeter or edges > blob_area * 2 or
                                         abs(box_area - avg_blob_area) > 7.5 * blob_area_std or abs(
                    edge_area_ratio - avg_edge_area_ratio) > 3 * edge_area_std or
                                         abs(f.aspectRatio() - avg_aspect_ratio) > 5.5 * aspect_ratio_std or
                                         f.aspectRatio() > 20 or
                                             3.8 * (f.perimeter() / 6.284) ** 2 <= blob_area or f.onImageEdge() or (
                        colors[0][0] + colors[1][0]) < 1.2 * sum(
                zip(*colors[2:])[0])):
                img.drawRectangle(x, y, w, h, color=(127, 0, 0), width=2)
                x1 = max(0, x - 8 * pad)
                y1 = max(0, y - pad)

                image_mask.drawRectangle(x1, y1, min(w + 16 * pad, image_mask.width - x1),
                                         min(h + 2 * pad, image_mask.height - y1), color=(255, 255, 255), width=-1)
            elif not (cropped.width > img.width * 0.8 or cropped.height > img.height * 0.8):
                filter_str = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (
                    abs(h - avg_height) > 5.5 * height_std, abs(w - avg_width) > 6.5 * width_std,
                    weird_blob > len(fs2) / 3.25, abs(w * h - avg_rec_area) > 4 * rec_area_std,
                    cropped.width > img.width * 0.9, cropped.height > img.height * 0.9,
                    edges < rec_perimeter, edges > blob_area * 2,
                    abs(box_area - avg_blob_area) > 7.5 * blob_area_std, abs(
                        edge_area_ratio - avg_edge_area_ratio) > 3 * edge_area_std,
                    abs(f.aspectRatio() - avg_aspect_ratio) > 5.5 * aspect_ratio_std,
                    f.aspectRatio() > 20,
                    3.8 * (f.perimeter() / 6.284) ** 2 <= blob_area, f.onImageEdge(),
                    (colors[0][0] + colors[1][0]) < 1.2 * sum(
                        zip(*colors[2:])[0]))
                logging.info(filter_str)
                img.drawText(filter_str, x, y)
                img.show()
                # raw_input("d")
            """else:
                if not (cropped.width > img.width * 0.8 or
                                cropped.height > img.height * 0.8):
                    img.drawRectangle(x, y, w, h, color=(127, 127, 0), width=-1)
                    img.drawText(
                            "%d%d%d%d%d%d%d%d%d%d%d%d%d%d" % (
                                weird_blob > len(fs2) / 2, cropped.width > img.width * 0.8,
                                cropped.height > img.height * 0.8,
                                edges < rec_perimeter,
                                edges > blob_area * 2,
                                abs(box_area - avg_blob_area) > 7 * blob_area_std, abs(
                                        edge_area_ratio - avg_edge_area_ratio) > 6 * edge_area_std,
                                f.aspectRatio() < 0.01, f.aspectRatio() > 70,
                                blob_area > box_area * 0.9, 3.15 * (
                                    f.perimeter() / 6.284) ** 2 <= blob_area,
                                f.onImageEdge(), colors[0][0] > 7 * colors[1][0],
                                colors[0][0] + colors[1][0] < 1.2 * sum(
                                        zip(*colors[2:])[0])), x, y)
                    logging.info("%d%d%d%d%d%d%d%d%d%d%d%d%d%d" % (weird_blob > len(fs2) / 2, cropped.width > img.width * 0.8,
                                                            cropped.height > img.height * 0.8,
                                                            edges < rec_perimeter,
                                                            edges > blob_area * 2,
                                                            abs(box_area - avg_blob_area) > 7 * blob_area_std, abs(
                            edge_area_ratio - avg_edge_area_ratio) > 6 * edge_area_std,
                                                            f.aspectRatio() < 0.01, f.aspectRatio() > 70,
                                                            blob_area > box_area * 0.9, 3.15 * (
                                                                f.perimeter() / 6.284) ** 2 <= blob_area,
                                                            f.onImageEdge(), colors[0][0] > 5 * colors[1][0],
                                                            colors[0][0] + colors[1][0] < 1.2 * sum(
                                                                    zip(*colors[2:])[0])))"""

            # img.show()
            # time.sleep(2)
            # f.crop().show()
            # time.sleep(2)
            it += 1
            # img.applyLayers()
        if it == len(fs):
            mask = image_mask.applyLayers()
            # time.sleep(1)
            # mask.show()
            # time.sleep(1)
    # img.show()
    # raw_input("df")
    """raw_input("d")
    img.clearLayers()
    rec = []
    gx = int(img.width ** 0.9)
    gy = int(img.height ** 0.9)
    for i in range(0, img.width, gx):
        for j in range(0, img.height, gy):
            cropped = img.crop(i, j, min(gx, img.width - i), min(gy, img.height - j))
            #cropped.show()
            #raw_input("h")
            contours_to_boxes(find_contours(cropped.binarize().invert()), rectangles=rec, max_width=cropped.width,
                                    max_height=cropped.height, x_pad=i, y_pad=j)
    logging.info(rec)
    for box in rec:
        img.drawRectangle(*box)
    img.show()"""
    if level == 0:
        kernel = np.ones((35, 45), np.uint8)

        blobs = Image(cv2.morphologyEx(image_mask.applyLayers().dilate(2).invert().getNumpy(), cv2.MORPH_OPEN,
                                       kernel)).invert().findBlobs()
        if blobs is None:
            return image_mask
        [image_mask.drawRectangle(*b.boundingBox(), color=(255, 255, 255), width=-1) for b in blobs]
    # image_mask.applyLayers().show()
    # img = image_mask.applyLayers().invert().erode(2) + (input_image + image_mask.applyLayers().invert().erode(2)).invert()
    return image_mask.applyLayers()


def morph_swt(swt):
    # swt = (swt.threshold(80) * 0.4 + 20) + swt
    lines = swt
    # swt.show()
    # raw_input("g")
    kernel1 = np.ones((25, 1), np.uint8)
    # swt.erode().dilate().erode().dilate().show()
    # raw_input("g")
    lines = Image(cv2.morphologyEx(swt.morphClose().getNumpy(), cv2.MORPH_OPEN, kernel1))
    kernel2 = np.ones((1, 6), np.uint8)
    hors = Image(cv2.morphologyEx(swt.morphClose().getNumpy(), cv2.MORPH_OPEN, kernel2))
    (lines + hors).show()
    do = (lines + hors).invert() + swt.blur().dilate().erode().morphClose().invert()
    # do = Image(cv2.morphologyEx(do.getNumpy(), cv2.MORPH_OPEN, np.ones((1, 6), np.uint8)))
    # do.gaussianBlur(window=(70, 1), sigmaX=8, sigmaY=8)
    # do.show()
    # raw_input("hoho")
    return do
    Image(cv2.morphologyEx(do.invert().getNumpy(), cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))).show()
    return lines + hors
    # lines.show()
    # raw_input("o1o")
    kernel = np.ones((30, 1), np.uint8)
    lines = frame_image(
        Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    lines = frame_image(
        Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    # lines.show()
    # raw_input("o2o")
    kernel = np.ones((1, 4), np.uint8)
    lines = frame_image(
        Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    return lines
    kernel = np.ones((4, 1), np.uint8)
    lines = frame_image(Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_CLOSE, kernel)), color=(255, 255, 255))
    kernel = np.ones((5, 1), np.uint8)
    lines = frame_image(Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)),
                        color=(255, 255, 255)).dilate()
    return lines


"""def filter_contours(image, swt, invert=True, min_size=5):
    # TODO research on invariant features for filtering
    img = frame_image(image)
    # img.show()
    area = image.width * image.height
    image_mask = Image((img.width, img.height))
    # logging.info(area)
    block_size = 2 * (int(area ** 0.7)) + 1
    # logging.info(block_size)

    pad = min(3, 1 + min(image.width, image.height) // 150)
    it = 0
    fs = img.findBlobs()  # threshblocksize=11, threshconstant=1, minsize=min_size)
    if fs is not None:
        heights = fs.height()
        avg_height = get_mean(heights)
        height_std = get_variance(heights, avg_height)
        fs = fs.filter(fs.height() < avg_height + 3 * height_std)
        blob_areas = fs.area()
        avg_blob_area = get_mean(blob_areas)
        blob_area_std = get_variance(blob_areas, avg_blob_area)
        fs = fs.filter(fs.area() < avg_blob_area + 4 * blob_area_std)

        crops = [swt.crop(*f.boundingBox()) for f in fs]
        edge_area_ratios = [float(crop.width * crop.height) /
                            (cv2.countNonZero(crop.morphGradient().getGrayNumpyCv2()) + 1) for crop in crops]
        avg_edge_area_ratio = get_mean(edge_area_ratios)
        edge_area_std = get_variance(edge_area_ratios, avg_edge_area_ratio)

        blob_areas = fs.area()
        avg_blob_area = get_mean(blob_areas)
        blob_area_std = get_variance(blob_areas, avg_blob_area)
        heights = fs.height()
        avg_height = get_mean(heights)
        height_std = get_variance(heights, avg_height)
        widths = fs.width()
        avg_width = get_mean(widths)
        width_std = get_variance(heights, avg_width)
        rec_areas = [w * h for w, h in zip(widths, heights)]
        avg_rec_area = get_mean(rec_areas)
        rec_area_std = get_variance(rec_areas, avg_rec_area)
        aspect_ratios = fs.aspectRatios()
        avg_aspect_ratio = get_mean(aspect_ratios)
        aspect_ratio_std = get_variance(aspect_ratios, avg_aspect_ratio)
        if len(fs) < 6:
            edge_area_std = 2 * (edge_area_std + 1) ** 2
            blob_area_std = 2 * (blob_area_std + 1) ** 2
            height_std = 2 * (height_std + 1) ** 2
            width_std = 2 * (width_std + 1) ** 2
        for f in fs:
            [x, y, w, h] = f.boundingBox()
            box_area = f.area()
            cropped = swt.crop(x, y, w, h)
            blob_area = cv2.countNonZero(cropped.getGrayNumpyCv2())
            if blob_area < 10:
                # logging.info("too small")
                continue
            colors = cropped.getGrayHistogramCounts(bins=4)
            # logging.info(colors)

            edges = cv2.countNonZero(cropped.morphGradient().getGrayNumpyCv2())
            edge_area_ratio = float(blob_area) / (edges + 1)
            bin_area = cv2.countNonZero(cropped.binarize().getGrayNumpyCv2())
            rec_perimeter = (cropped.width + cropped.height)
            fs2 = swt.crop([max(0, x - 2), max(0, y - 2), min(w + 4, swt.width - x + 2),
                            min(h + 4, swt.height - y + 2)]).findBlobs()
            weird_blob = 0
            if fs2 == None:
                # img.drawRectangle(x, y, w, h, color=(127, 0, 0), width=-1)
                # logging.info("no blobs inside")
                continue
            for f2 in fs2:
                if f2.aspectRatio() < 0.1 or f2.aspectRatio() > 4:
                    weird_blob += 1
            if not (abs(h - avg_height) > 2 * height_std or abs(h-avg_width)> 3*width_std or abs(
                            w * h - avg_rec_area) > 2*rec_area_std or weird_blob > len(
                    fs2) / 2 or cropped.width > img.width * 0.8 or cropped.height > img.height * 0.8 or
                                    edges < rec_perimeter or edges > blob_area * 3 or
                                    abs(box_area - avg_blob_area) > 1.8 * blob_area_std or abs(
                        edge_area_ratio - avg_edge_area_ratio) > 1.8 * edge_area_std or f.aspectRatio() < 0.01 or
                                    f.aspectRatio() > 15 or abs(
                        f.aspectRatio() - avg_aspect_ratio) > aspect_ratio_std or blob_area > w * h * 0.9 or
                                        3.15 * (f.perimeter() / 6.284) ** 2 <= blob_area or f.onImageEdge() or
                                    colors[0][0] > 7 * colors[1][0] or (colors[0][0] + colors[1][0]) < 1.2 * sum(
                    zip(*colors[2:])[0])):
                img.drawRectangle(x, y, w, h, color=(127, 0, 0), width=2)
                x1 = max(0, x - 8 * pad)
                y1 = max(0, y - pad)
                # logging.info("image_mask%d,%d,%d,%d" % (x1, y1, min(w + 10 * pad, image_mask.width - x1),
                #                                 min(h + 2 * pad, image_mask.height - y1)))
                image_mask.drawRectangle(x1, y1, min(w + 16 * pad, image_mask.width - x1),
                                         min(h + 2 * pad, image_mask.height - y1), color=(255, 255, 255), width=-1)
        it += 1
        if it == len(fs):
            mask = image_mask.applyLayers()
    if invert:
        img = img.invert()
    kernel = np.ones((35, 45), np.uint8)
    blobs = Image(cv2.morphologyEx(image_mask.applyLayers().dilate(2).invert().getNumpy(), cv2.MORPH_OPEN,
                                   kernel)).invert().findBlobs()
    if blobs is None:
        return image_mask.invert()
    [image_mask.drawRectangle(*b.boundingBox(), color=(255, 255, 255), width=-1) for b in blobs]
    return image_mask.applyLayers()


def morph_swt(swt):
    # swt = frame_image(swt.threshold(100))
    return swt.invert()

    kernel = np.ones((3, 1), np.uint8)
    lines = Image(cv2.morphologyEx(swt.getNumpy(), cv2.MORPH_CLOSE, kernel))
    lines.show()
    kernel = np.ones((8, 1), np.uint8)
    lines = frame_image(
            Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    lines = frame_image(
            Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    kernel = np.ones((1, 4), np.uint8)
    lines = frame_image(
            Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    kernel = np.ones((4, 1), np.uint8)
    lines = frame_image(Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_CLOSE, kernel)), color=(255, 255, 255))
    kernel = np.ones((5, 1), np.uint8)
    lines = frame_image(Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)),
                        color=(255, 255, 255)).dilate().erode().dilate()
    lines.show()
    # raw_input("dodo")
    return lines
def morph_swt(swt):
    swt = (swt.threshold(80) * 0.4 + 20) + swt
    lines = swt
    return swt.blur().dilate()
    kernel = np.ones((3, 1), np.uint8)
    # swt.erode().dilate().erode().dilate().show()
    # raw_input("g")
    lines = Image(cv2.morphologyEx(swt.erode().dilate().getNumpy(), cv2.MORPH_CLOSE, kernel))
    lines.show()
    # raw_input("o1o")
    kernel = np.ones((8, 1), np.uint8)
    lines = frame_image(
            Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    lines = frame_image(
            Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    # lines.show()
    # raw_input("o2o")
    kernel = np.ones((1, 4), np.uint8)
    lines = frame_image(
            Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)), color=(255, 255, 255))
    kernel = np.ones((4, 1), np.uint8)
    lines = frame_image(Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_CLOSE, kernel)), color=(255, 255, 255))
    kernel = np.ones((5, 1), np.uint8)
    lines = frame_image(Image(cv2.morphologyEx(lines.getNumpy(), cv2.MORPH_OPEN, kernel)),
                        color=(255, 255, 255)).dilate().erode().dilate()
    return lines"""


def test_filter_contours(image, dob=0, skip=True):
    if not skip:
        adaptive_histeq(image, "temp1.png")
        subprocess.call(['data/external_SWT/./DetectText',
                         "temp1.png", 'result.png', '%d' % dob])
    swt = Image("SWT.png")
    lines = morph_swt(swt)
    lines.show()
    raw_input("h")
    filtered = filter_contours(lines, swt.invert(), invert=True, min_size=3)
    filtered.show()
    raw_input("dodo")


# test_filter_contours("data/benchmark_images/butler_garage.png",dob=1)

def find_cut_indexes(hor, threshold):
    cumulative_sum = CumulativeSum()
    if len(hor) < 2:
        return []
    hor_cum_sum = [cumulative_sum(i) for i in hor]
    hor_grad = gradient_1d(hor_cum_sum)
    hor_grad = gradient_1d(hor_grad)
    gaps = []
    gap_started = False
    max_gap = -1
    cuts = []
    for i, j in enumerate(hor_grad):
        if j <= 0:
            if not gap_started and hor[i] <= threshold:
                gap_started = True
                gap_start_index = i
                cuts.append(i)
        else:
            if gap_started:
                gap_started = False
                gap_length = i - gap_start_index
                cuts.append(max(0, i - 2))
                if max_gap <= gap_length:
                    max_gap = gap_length
                gaps.append((gap_start_index, i))
    if gap_started:
        gap_started = False
        gaps.append((gap_start_index, len(hor_grad) - 1))
    if len(gaps) == 0:
        return []
    return cuts


def horizontal_close(img, morph=cv2.MORPH_CLOSE):
    lines = img
    kernel = np.ones((min(7, (img.width + 4) // 2), 1), np.uint8)
    lines = Image(
        cv2.morphologyEx(img.invert().getNumpy(), morph, kernel)).erode().dilate().erode(3).invert()
    return lines


def vertical_morph(img, morph=cv2.MORPH_CLOSE):
    lines = img
    kernel = np.ones((1, min(7, (img.width + 4) // 2)), np.uint8)
    lines = Image(
        cv2.morphologyEx(img.invert().getNumpy(), morph, kernel)).dilate().erode().dilate(3).invert()
    return lines


def xy_cut(x, y, width, height, image, nump, horizontal=True):
    if height <= 3 or width <= 3:
        return
    if horizontal:
        hor = [sum([nump[x + i, y + j][0] for i in range(0, width)]) for j in
               range(0, height)]  # [sum(nump[x:x + width, y + i][0]) for i in range(0, height)]
        # logging.info(hor)
        aspect = float(height) / width
        ys = find_cut_indexes(hor, width * aspect * 20)
        prev_y = y
        for new_y in ys:
            if new_y - (prev_y - y) < 5:  # >= height - 5 :
                continue
            image.drawLine((x, y + new_y), (x + width - 1, y + new_y), color=(0, 255, 0), thickness=3)
            xy_cut(x, prev_y, width, new_y - (prev_y - y), image, nump, horizontal=False)
            prev_y = y + new_y
        if prev_y != y:
            xy_cut(x, prev_y, width, y + height - prev_y, image, nump, horizontal=False)
    else:
        ver = [sum([nump[x + i, y + j][0] for j in range(0, height)]) for i in range(0, width)]
        aspect = float(width) / height
        xs = find_cut_indexes(ver, height * aspect)
        prev_x = x
        for new_x in xs:
            if new_x - (prev_x - x) < 5:
                continue
            image.drawLine((new_x + x, y), (new_x + x, y + height - 1), color=(0, 255, 0), thickness=3)
            # logging.info(prev_x, y, new_x, height)
            xy_cut(prev_x, y, new_x - (prev_x - x), height, image, nump, horizontal=True)
            prev_x = x + new_x
        if prev_x != x:
            xy_cut(prev_x, y, x + width - prev_x, height, image, nump, horizontal=True)


def img_to_1d_gray(img):
    """convert image into 1D gray values
    @return 1d np array"""
    if isinstance(img, basestring):
        img = Image(img)
    elif not hasattr(img, 'toGray'):
        logging.error('img type is neither string nor image')
    return img.toGray().getNumpy()[:, :, 0].reshape(img.height * img.width)


def img_from_csv_line(img, height, width, label_count=1):
    return Image(np.array([int(i) for i in img.strip().split(",")[label_count:]]).reshape(height, width))


def random_invert_crop(img, full=True):
    if not full:
        x = random.randint(0, img.width // 2)
        y = random.randint(0, img.height // 2)
        w = random.randint(1, img.width - 1 - x)
        h = random.randint(1, img.width - 1 - y)
        new_im = img.crop(x, y, w, h).invert()
        new_im = img.blit(new_im, (x, y)).applyLayers()
    else:
        new_im = img.invert()
    return new_im


def img_to_csv_line(image):
    return ",".join(str(x) for x in img_to_1d_gray(image)) + "\n"


def csv_to_img(csv_name, label_count=1):
    """@param csv_name: of a file that contains labels in its first columns and then pixel values"""
    with open(csv_name, "r") as csv_file:
        lines = [img for img in csv_file][1:]
        assert (len(lines) > 0)
        # logging.info(len(lines[0]) - label_count)
        height = width = sqrt(len(lines[0].strip().split(",")) - label_count)
        for img in lines:
            # image = img_from_csv_line(img, label_count, height, width)
            # image = random_invert_crop(image)
            # csv_file.write(img_to_csv_line(image))
            Image(np.array([int(i) for i in img.strip().split(",")[label_count:]]).reshape(height, width)).show()


def map_from_mapper(y_train, mapper, left=True):
    for i, row in enumerate(y_train):
        for j, val in enumerate(row):
            if left:
                if mapper.left_to_right[y_train[i][j]] is None:
                    y_train[i][j] = mapper.left_to_right['*']  # default don't know class
                else:
                    y_train[i][j] = mapper.left_to_right[y_train[i][j]]
            else:
                '''TODO right to left can be many to one, which one to train? First one?'''
                y_train[i][j] = mapper.right_to_left[y_train[i][j]][0]
    return y_train


def map_from_input_to_integer(y_train, nb_classes):
    with open("integer_labels.csv", "w") as integer_labels_csv:
        if y_train is None or len(y_train) < 1:
            logging.warning("y_train is empty")
            return y_train
        explored_index = 0
        if len(y_train[0]) < 1:
            logging.warning("y_train rows are empty")
            return y_train
        d = {y_train[0][0]: explored_index}
        for i, row in enumerate(y_train):
            for j, val in enumerate(row):
                if d.has_key(val):
                    y_train[i][j] = d[val]
                else:
                    explored_index += 1
                    assert (explored_index < nb_classes)
                    d[val] = explored_index
                    y_train[i][j] = explored_index
        for key in d:
            integer_labels_csv.write(str(key) + "," + str(d[key]) + "\n")


def find_all(search_pattern, flags="-type f", path="`pwd`"):
    """ recursively search for a pattern in all file or directory names
    @param search_pattern: must be a regular expression for grep command:
        e.g. "[.]py$", "[.][^/,^.]\\+$", ".txt$", "[^.]\\+/test"
    @param flags: e.g. -maxtdepth 1
    @param path: full path to search under
    @rtype: list
    @return full paths of files or directories that fit the search pattern
        [] if none found
    @todo support mac and windows too"""
    return subprocess.Popen("find %s %s | grep '%s'" % (path, flags, search_pattern), shell=True, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE).stdout.read().strip().split('\n')


def float_or_nan(str_float):
    """convert @type basestring to @type float
    @return NaN if failed"""
    try:
        a = float(str_float)
    except ValueError:
        a = "__NaN__"
    return a


def int_or_nan(str_int):
    """convert string to int

    if string is convertible to float
    convert to float then int    
    return NaN if failed"""
    try:
        a = int(str_int)
    except ValueError:
        try:
            a = int(float(str_int))
        except ValueError:
            a = "__NaN__"
    return a


def empty_to_mode(column, random_select_from_modes=True):
    """change empty entries in the column to the mode value"""
    arr = [x for x in column if not x == '']
    categories = {i: 0 for i in arr}
    mode_max = 0
    modes = []
    for (i, j) in enumerate(arr):
        categories[j] += 1
        if mode_max < categories[j]:
            mode_max = categories[j]
            mode_val = j
            modes = [mode_val]
        elif mode_max == categories[j]:
            modes.append(j)
    if random_select_from_modes is False or len(modes) == 1:
        return [mode_max if x == "" else x for x in column]
    return [modes[random.randint(0, len(modes) - 1)] if x == "" else x for x in column]


def vertical_split(list_of_lists, split_index):
    input_rows = [values[split_index:]
                  for values in list_of_lists]
    output_rows = [values[:split_index]
                   for values in list_of_lists]
    assert (len(output_rows) == len(input_rows))
    return input_rows, output_rows


def nan_to_mean(column, string_handler):
    arr = [x for x in column if not isinstance(x, basestring)]
    if len(arr) == 0:  # all string
        if column[0] == "__NaN__":  # all of it has __NaN__ strings
            return [0] * len(column)
        return string_handler(column)
    if len(arr) == len(column):  # all numbers
        return column
    avg_val = sum(arr) / len(column)
    return [avg_val if x == "__NaN__" else x for x in column]


def load_csv(fname_csv, data_row=1):
    with open(fname_csv) as csv_f:
        file_data = [line.strip().split(",") for line in csv_f]
        labels = file_data[data_row - 1:data_row]
        data = file_data[data_row:]
        return labels, data


def fill_missing_data(dataset, column_id=0, number_handler=nan_to_mean, string_handler=empty_to_mode):
    """Handle NaN entries and empty strings given a column id, update the given data
    @param dataset: a list of lists corresponding to a matrix
    @param number_handler: convert a given column with NaN entries to one without, 
        call string_handler if given column is a string column and does not contain NaN
    @param string_handler: convert a given column with empty strings to one without
    """
    column = [row[column_id] for row in dataset]
    new_column = number_handler(column, string_handler)
    for i, j in enumerate(new_column):
        dataset[i][column_id] = j


'''def plot_confusion_matrix(cm=None, labels=None, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')'''


def remap_csv_labels(remapper, input_csv, output_csv):
    with open(input_csv, "r") as old_csv, open(output_csv, "w") as new_csv:
        for line in input_csv:
            elements = line.strip().split(",")
            if len(elements) < 1:
                continue
            new_csv.write(str(remapper[elements[0]]) + "".join(elements[1:]) + "\n")


def label_remapper():
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
              'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
              'y', 'z', ]


def init_mapper(mapper):
    i = 1
    for k in string.digits + string.ascii_letters:
        # reduce classes
        # logging.info(k,i)
        if k == 'O' or k == 'o':
            mapper[k] = '1'
        elif k == 'i' or k == 'l' or k == 'I':
            mapper[k] = '2'
        elif k == 'g':  # '9' -> '10' & 'g' -> '10'
            mapper[k] = '10'
        elif k == 'C':
            mapper[k] = '13'
        elif k == 'J':
            mapper[k] = '18'
        elif k == 'P':
            mapper[k] = '22'
        elif k == 'S':
            mapper[k] = '25'
        elif k == 'V':
            mapper[k] = '28'
        elif k == 'W':
            mapper[k] = '29'
        elif k == 'M':
            mapper[k] = '20'
        elif k == 'U':
            mapper[k] = '27'
        elif k == 'X':
            mapper[k] = '30'
        elif k == 'Z':
            mapper[k] = '32'
        else:
            mapper[k] = str(i)
            i += 1
    mapper['*'] = '0'  # the rest is 0


class Mapper:
    """for mapping of hashable objects"""

    def __init__(self):
        self.left_to_right = {}
        self.right_to_left = {}

    def __getitem__(self, key):
        if key in self.left_to_right:
            if key in self.right_to_left:
                return self.right_to_left[key], self.right_to_left[key]
            else:
                return self.left_to_right[key], None
        elif key in self.right_to_left:
            return None, self.right_to_left[key]
        else:
            return None, None

    def __setitem__(self, key, value):
        """many to one, left to right,
        one to many, right to left"""
        self.left_to_right[key] = value
        if value in self.right_to_left:
            self.right_to_left[value].append(key)
        else:
            self.right_to_left[value] = [key]

    def remove(self, k):
        self.right_to_left.pop(self.left_to_right.pop(k))

    def get(self, k):
        return self.__getitem__(k)


class RectangleElement:
    def __init__(self, char=None, rectangle=None, feature=None):
        self.char = char
        self.rectangle = rectangle
        self.feature = feature

    def __lt__(self, other):
        # upper line
        if self.rectangle[1] + self.rectangle[3] < other.rectangle[1]:
            return True
        if other.rectangle[1] + other.rectangle[3] < self.rectangle[1]:
            return False

        if self.rectangle[0] < other.rectangle[0] + other.rectangle[2] / 2.0 and self.rectangle[1] + self.rectangle[
            3] / 2.0 <= other.rectangle[1] + other.rectangle[3]:
            return True
        return False

    def __gt__(self, other):
        return not self.lt(other)


def compile_into_text(chars):
    """create a text from ``chars`` according to their positions

    :param chars: A list of rectangles of the form :class `RectangleElement`.
    :rtype: type(sentences) = A `list <list <RectangleElements> >`
    """
    if len(chars) < 1:
        return ''
    assert (hasattr(chars[0], 'rectangle'))
    sentences = []
    sentence = []
    word = ""
    previous = None
    previous_space = 100
    for rectangle_char in sorted(chars):
        if previous is None:
            assert (len(word) == 0)
            word += rectangle_char.char
        else:
            if rectangle_char.rectangle[1] + rectangle_char.rectangle[3] / 2 > previous.rectangle[1] + \
                    previous.rectangle[3]:
                # end of a sentence/line, start a new sentence
                sentence.append(word)
                sentences.append(sentence)
                sentence = []
                word = rectangle_char.char
            elif rectangle_char.rectangle[0] - (
                        previous.rectangle[0] + previous.rectangle[2]) > 1 + previous_space * 1.2 + (
                        rectangle_char.rectangle[2] + previous.rectangle[2]) * 0.05:
                # end of a word, start a new word
                sentence.append(word)
                word = "" + rectangle_char.char
            else:
                word += rectangle_char.char
            previous_space = max(rectangle_char.rectangle[0] - (previous.rectangle[0] + previous.rectangle[2]), 0)
        previous = rectangle_char

    sentence.append(word)
    sentences.append(sentence)
    return sentences


def show_bad_fonts():
    i = Image((400, 400)).invert()
    for k in i.dl().listFonts():
        i.clearLayers()
        i.dl().selectFont(k)
        i.drawText(k)
        i.show()
        m = k
        raw_input(m)


def find_rectangle_centroid(rec):
    """rec=[x,y,width,height]
    returns (x_c,y_c) """
    return rec[0] + rec[2] / 2, rec[1] + rec[3] / 2


def horizontal_rotate(image):
    angle, new_image = smart_rotate(image)
    logging.info(image.width, image.height)
    new_image = crop_padding(new_image)
    logging.info(new_image.width, new_image.height, angle)
    new_image.show()
    # raw_input("smart_rotate_crop")
    if new_image.width * 1.4 < image.width:  # vertical
        new_image = new_image.rotate(-90, fixed=False, )
        angle = -90
    return angle, crop_padding(new_image)


def smart_rotate(image, bins=18, point=[-1, -1], auto=True, threshold=80, minLength=30, maxGap=10, t1=150, t2=200,
                 fixed=False):
    '''This is a modified SimpleCV method'''
    lines = image.findLines(threshold, minLength, maxGap, t1, t2)
    if len(lines) == 0:
        logging.warning("No lines found in the image")
        return image
    binn = [[] for i in range(bins)]
    conv = lambda x: int(x + 90) / bins
    [binn[conv(line.angle())].append(line) for line in lines]
    hist = [sum([line.length() for line in lines]) for lines in binn]
    index = np.argmax(np.array(hist))
    avg = sum([line.angle() * line.length() for line in binn[index]]) / sum([line.length() for line in binn[index]])
    if (auto):
        x = sum([line.end_points[0][0] + line.end_points[1][0] for line in binn[index]]) / 2 / len(binn[index])
        y = sum([line.end_points[0][1] + line.end_points[1][1] for line in binn[index]]) / 2 / len(binn[index])
        point = [x, y]
    if -45 <= avg <= 45:
        return avg, image.rotate(avg, fixed=fixed, point=point)
    elif 90 > avg > 45:
        return avg - 90, image.rotate(avg - 90, fixed=fixed, point=point)
    elif -90 < avg < -45:
        return avg + 90, image.rotate(avg + 90, fixed=fixed, point=point)
    else:
        return 0, image


def crop_padding(image):
    ver = image.verticalHistogram(bins=image.height)
    hor = image.horizontalHistogram(bins=image.width)
    left = 0
    right = image.width
    top = 0
    bottom = image.height
    k = 0
    for i in hor:
        if i != 0:
            left = k
            break
        k += 1
    k = image.width - 1
    for i in hor[::-1]:
        if i != 0:
            right = k
            break
        k -= 1
    k = 0
    for j in ver:
        if j != 0:
            top = k
            break
        k += 1
    k = image.height - 1
    for j in ver[::-1]:
        if j != 0:
            bottom = k
            break
        k -= 1
    return image.crop((left, top), (right, bottom))


def key_compare(char1, char2):
    if char1.rectangle[0] > char2.rectangle[0]:
        return char1.rectangle[1] + char1.rectangle[3] < char2.rectangle[1]
    else:
        return char1.rectangle[1] >= char2.rectangle[1] + char2.rectangle[3]


def after_label_or_zero(text, label, reverse=False):
    if reverse:
        start_i = text.rfind(label)
    else:
        start_i = text.find(label)
    if start_i < 0:
        return 0
    return start_i + len(label)


def before_label_or_zero(text, label, reverse=False):
    if reverse:
        start_i = text.rfind(label)
    else:
        start_i = text.find(label)
    if start_i < 0:
        return 0
    return start_i


def find_between_labels(text, first_label, last_label, scheme='01', around_first=True):
    """ find the slice of text that is between the given labels

    :param text: input string
    :param first_label:
    :param last_label:
    :param scheme: '01', '10', '00' or '11', (e.g. 01: reverse search for first label, normal for second)
    :return: string found between the labels or empty string
    """
    if first_label is None:
        first_label = ""
    if last_label is None:
        last_label = ""
    if text is None:
        return ""
    if not isinstance(first_label, basestring):
        first_label = str(first_label)
    if not isinstance(last_label, basestring):
        last_label = str(last_label)
    if around_first:
        if scheme[0] == '1':
            start_i = after_label_or_zero(text, first_label)
        else:  # scheme[0]=='l'
            start_i = after_label_or_zero(text, first_label, reverse=True)
        if scheme[1] == '1':
            end_i = before_label_or_zero(text[start_i:], last_label) + start_i
        else:  # scheme[1]=='l'
            end_i = before_label_or_zero(text, last_label, reverse=True)
    else:  # search around the last label
        if scheme[1] == '1':
            end_i = before_label_or_zero(text, last_label)
        else:  # scheme[1]=='l'
            end_i = before_label_or_zero(text, last_label, reverse=True)
        if scheme[0] == '1':
            start_i = after_label_or_zero(text, first_label)
        else:  # scheme[0]=='l'
            start_i = after_label_or_zero(text[:end_i], first_label, reverse=True)
    if end_i < start_i:
        logging.warn(
            "Can only find a reverse interval from %s to %s: %s" % (end_i, start_i, text[start_i:end_i].strip()))
        return ""
    return text[start_i:end_i].strip()


def list3d_to_df(list3d):
    """[[],..],[[],..]"""
    import pandas as pd
    logging.info(pd.DataFrame(data=list3d))


def get_all_files(input_folder, check=lambda f: f[-4:] == '.mp4'):
    import glob
    files = glob.glob(input_folder)
    logging.info(files)
    return [f for f in files if len(f) > 4 and check(f)]


def move_files(folder_dict, path):
    """move files in a given path to the target folders there, according to the dictionary
    :param folder_dict: {folder_name1:[file1,file2]}
    :param path: files will be searched and moved into the folders which will be created under this path
    """
    import shutil
    for key, val in folder_dict.items():
        logging.info(key)
        if not os.path.isdir(path + key):
            os.mkdir(path + key)
        for i in val:
            logging.info(i, path + key + "/" + get_file_name(i, no_extension=False))
            shutil.move(i, path + key + "/" + get_file_name(i, no_extension=False))


def get_file_name(input_file, no_extension=True):
    """get only the file name and no extension by default, if no_extension is False get the full file name"""
    if no_extension:
        return input_file[input_file.rfind("/") + 1:input_file.rfind(".")]
    else:
        return input_file[input_file.rfind("/") + 1:]


def convert_xs(xs, converter=lambda x: float(x[:-1])):
    return [converter(x) for x in xs]


def expand_args(x, *xs):
    return tuple([x]) + xs


def f_or_default(arg, default=None, func=lambda x: x[0](*x[1:])):
    """if given argument is None return the default value, else return func(arg)"""
    if arg is None:
        return default
    else:
        return func(arg)


def mutate_list(random_vals, low=0, high=0, replace=np.random.uniform):
    mutate_index = np.random.randint(0, len(random_vals))
    logging.info("old list:")
    logging.info(random_vals)
    random_vals[mutate_index] = replace(low=low, high=high)
    logging.info("mutated into:")
    logging.info(random_vals)
    return random_vals

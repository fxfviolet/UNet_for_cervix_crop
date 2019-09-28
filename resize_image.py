import json
import ntpath
import os

import cv2
import numpy as np
import glob

FILE_PATTERN = '*.jpg'

def resize_testset(source_folder, target_folder, dsize, pattern=FILE_PATTERN):
    print('Resizing testset ...')
    if not os.path.exists(target_folder): os.makedirs(target_folder)
    total_images = glob.glob(os.path.join(source_folder, pattern))
    total = len(total_images)
    for i, source in enumerate(total_images):
        filename = ntpath.basename(source)
        target = os.path.join(target_folder, filename.replace('.jpg', '.png'))

        img = cv2.imread(source)
        img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(target, img_resized)
        if i % 100 == 0:
            print("Resized {}/{} images".format(i, total))


if __name__ == '__main__':
    input_folder = 'E:/Unet/cervix-roi-segmentation-by-unet/input/Hsil'
    resized_folder = 'E:/Unet/cervix-roi-segmentation-by-unet/input/Hsil_resized'

    img_width = 512
    img_height = 512
    resize_testset(input_folder, resized_folder, (img_width, img_height))

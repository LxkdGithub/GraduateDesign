import cv2
import os
import numpy as np
import random
import argparse

mean = 0
var = 0.001

def Process(img_path, begin, end):
    images = os.listdir(img_path)
    images.sort(key=lambda x: (int(x[:5])*1000 + int(x[6:12]), x[13]))  # 00001-000136-0.png
    abs_path = os.path.abspath(img_path) + "/"
    new_abs_path = abs_path.replace("crop", "crop_aug") + "/"
    if not os.path.exists(new_abs_path):
        os.makedirs(new_abs_path)
    i = 1
    for image in images:
        if i < begin:
            i += 1
            continue
        if i > end:
            break
        print("------  {}  -----------".format(i))
        image_name = os.path.splitext(image)[0]
        print(image_name)
        img = cv2.imread(abs_path+image, cv2.IMREAD_GRAYSCALE)
        random_size = random.randrange(360, 720, 20)
        zoomed = cv2.resize(img, (random_size, random_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(new_abs_path+image_name+"-1.png", zoomed)

        center = (img.shape[1]//2, img.shape[0]//2)
        rotation = cv2.getRotationMatrix2D(center, 90*random.randint(1,3), 1)
        rotated = cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
        cv2.imwrite(new_abs_path+image_name+"-2.png", rotated)

        img_arr = np.array(img/255, dtype=float)
        noise = np.random.normal(mean, var**0.7, img.shape)
        noised = img_arr + noise
        if noised.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        noised = np.clip(noised, low_clip, 1.0)
        noised = np.uint8(noised * 255)
        cv2.imwrite(new_abs_path+image_name+"-3.png", noised)

        i += 1


if __name__ == "__main__":
    forged_img_path = "../images/train/forged_crop"
    prist_img_path = "../images/train/prist_crop"

    parser = argparse.ArgumentParser("Get Position")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()
    begin = args.begin
    end = args.end
    Process(forged_img_path, begin, end)
    Process(prist_img_path, begin, end)
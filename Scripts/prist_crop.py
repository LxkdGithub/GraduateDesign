import os
import cv2
import argparse


def train_split(begin, end, isTrain):
    if isTrain:
        t_img_path = "../images/train/prist/"
        crop_img_path = "../images/train/prist_crop/"
    else:
        t_img_path = "../images/test/prist/"
        crop_img_path = "../images/test/prist_crop/"
    if not os.path.exists(crop_img_path):
        os.mkdir(crop_img_path)
    img_list = os.listdir(t_img_path)
    img_list.sort(key=lambda x: int(x[:5] + x[6:12]))
    idx = 1
    for img_name in img_list:
        if idx < begin:
            idx += 1
            continue
        if idx > end:
            break
        crop_idx = 0
        prist = cv2.imread(t_img_path + img_name, cv2.IMREAD_GRAYSCALE)
        for i in range(3):
            roi = prist[:, i*280:i*280+720].copy()
            new_name = img_name[:-4] + '-' + str(crop_idx) + ".png"
            cv2.imwrite(crop_img_path + new_name, roi)
            crop_idx += 1
        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get Position")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    begin = args.begin
    end = args.end
    # 在其他脚本调用
    if not args.test:
        train_split(begin, end, True)
    else:
        train_split(begin, end, False)


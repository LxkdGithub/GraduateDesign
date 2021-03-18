import os
import cv2
import argparse


def split(begin, end, isTrain):
    if isTrain:
        t_img_path = "../images/train/forged/"
        crop_img_path = "../images/train/forged_crop/"
        write_file = "../images/train/pos.txt"
    else:
        t_img_path = "../images/test/forged/"
        crop_img_path = "../images/test/forged_crop/"
        write_file = "../images/test/pos.txt"

    if not os.path.exists(crop_img_path):
        os.mkdir(crop_img_path)
    idx = 1
    with open(write_file, "r") as pos_file:
        line = pos_file.readline()
        while line is not None and line != "":
            if idx < begin:
                line = pos_file.readline()
                idx += 1
                continue
            if idx > end:
                break
            crop_idx = 0
            line_split = line.split(':')
            img_name = line_split[0]
            coordinate = line_split[1].split('-')
            x_left = int(coordinate[0])
            x_right = int(coordinate[2])
            # Debug
            #
            forged = cv2.imread(t_img_path + img_name, cv2.IMREAD_GRAYSCALE)
            forged_color = cv2.imread(t_img_path + img_name)
            crop_count = 0
            if x_left < 80:
                print("<80")
                roi = forged[:, 0:720].copy()         # 尽量往左一点 因为本身就很靠左了 右边不会有误差 所以从0开始
                new_name = img_name[:-4] + '-' + str(crop_idx) + ".png"
                cv2.imwrite(crop_img_path + new_name, roi)
                crop_count += 1
            elif x_right > 1200:
                print(">1200")
                roi = forged[:, 560:].copy()
                new_name = img_name[:-4] + '-' + str(crop_idx) + ".png"
                cv2.imwrite(crop_img_path + new_name, roi)
                crop_count += 1
            else:
                x_l = max(x_right-720, 0)
                left_begin = x_l + 3
                while x_l < x_left and (x_l + 720 >= x_right) and (x_l < 560):
                    roi = forged[:, x_l:x_l+720].copy()
                    new_name = img_name[:-4] + '-' + str(crop_idx) + ".png"
                    cv2.imwrite(crop_img_path + new_name, roi)
                    crop_idx += 1
                    x_l += 30
                    crop_count += 1
                right_end = x_l + 687
                cv2.namedWindow("forged", 0)
                Width = forged.shape[1] // 2
                Height = forged.shape[0] // 2
                # cv2.resizeWindow("forged", Width, Height)
                cv2.moveWindow("forged", 100, 300)
                cv2.line(forged_color, (left_begin, 0), (left_begin, 720), (0, 255, 0), 1, 4)
                cv2.line(forged_color, (right_end, 0), (right_end, 720), (0, 255, 0), 1, 4)
                cv2.rectangle(forged_color, (x_left, 100), (x_right, 620), (0, 0, 255), 2)
                # print(left_begin, right_end)
                print("count: ", crop_count)
                # cv2.imshow("forged", forged_color)
                # cv2.waitKey(300)

            line = pos_file.readline()
            idx += 1


def test_forged_split(begin, end):
    t_img_path = "../images/test/forged/"
    crop_img_path = "../images/test/forged_crop/"
    if not os.path.exists(crop_img_path):
        os.mkdir(crop_img_path)
    img_list = os.listdir(t_img_path)
    img_list.sort(key=lambda x: int(x[:5]))
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
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    begin = args.begin
    end = args.end
    # 在其他脚本调用
    if args.test == 0:
        split(begin, end, True)
    else:
        # split(begin, end, False)
        test_forged_split(begin, end)

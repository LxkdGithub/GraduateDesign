import cv2
import os
import argparse


def get_pos(forged_path, origin_path, file):
    origin = cv2.imread(origin_path)
    forged = cv2.imread(forged_path)
    (_, img_name) = os.path.split(forged_path)
    origin_gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    forged_gray = cv2.cvtColor(forged, cv2.COLOR_BGR2GRAY)
    # delta -> threshold
    delta = cv2.subtract(forged_gray, origin_gray)
    ret, delta_ = cv2.threshold(delta, 7, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(delta_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if x > 50:
            x -= 30
        w += 60
        if w < 200:
            if x > 100:
                padding = (200 - w) // 2
                x -= padding
            w = 200
        if h < 35:
            continue

        # ----------  DEBUG ------------
        # roi = origin[y:y + h, x:x + w].copy()
        # draw_rec_prist = cv2.rectangle(origin, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # draw_rec_forged = cv2.rectangle(forged, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Width = origin.shape[1] // 2
        # Height = origin.shape[0] // 2
        # cv2.namedWindow('origin', 0)
        # cv2.namedWindow('forged', 0)
        # cv2.namedWindow("roi", 0)
        # cv2.resizeWindow('origin', Width, Height)
        # cv2.resizeWindow('forged', Width, Height)
        # cv2.resizeWindow('roi', w, h)
        # cv2.moveWindow("origin", 50, 50)
        # cv2.moveWindow("forged", 1000, 50)
        # cv2.moveWindow("roi", 700, 50)
        # cv2.imshow("origin", draw_rec_prist)
        # cv2.imshow("forged", draw_rec_forged)
        # cv2.imshow("roi", roi)
        print(w, h)
        # -------------------- DEBUG ----------------

        file.writelines("{}:{}-{}-{}-{}\n".format(img_name, x, y, x+w, y+h))
    # cv2.waitKey(100)


def all_get_pos(begin, end, isTrain):
    if isTrain:
        forged_path = os.path.abspath("../images/train/forged")
        origin_path = os.path.abspath("../images/train/prist")
        write_file = "../images/train/pos.txt"
    else:
        forged_path = os.path.abspath("../images/test/forged")
        origin_path = os.path.abspath("../images/test/prist")
        write_file = "../images/test/pos.txt"

    forged_imgs = os.listdir(forged_path)
    forged_imgs.sort(key=lambda x: int(x[:5]))
    i = 1
    with open(write_file, "a+") as write_file:
        for img in forged_imgs:
            if i < begin:
                i += 1
                continue
            if i > end:
                break
            get_pos(os.path.join(forged_path, img), os.path.join(origin_path, img), write_file)
            i += 1
            if i % 10 == 0:
                write_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get Position")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()
    begin = args.begin
    end = args.end
    # 函数调用如下(在其他脚本调用)
    if args.test == 0:
        all_get_pos(begin, end, True)
    else:
        all_get_pos(begin, end, False)



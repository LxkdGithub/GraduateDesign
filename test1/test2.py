import cv2
import os
import multiprocessing as mp

def test1():
    origin1 = cv2.imread("../images/valid/prist/00002-000081.png")
    origin2 = cv2.imread("../images/valid/prist/00002-000082.png")
    forged1 = cv2.imread("../images/valid/forged/00002-000081.png")
    forged2 = cv2.imread("../images/valid/forged/00002-000082.png")
    delta1 = cv2.subtract(origin1, origin2)
    delta2 = cv2.subtract(forged1, forged2)
    # cv2.imshow("delta1", delta1)
    # cv2.imshow("delta2", delta2)
    delta3 = cv2.subtract(delta1, delta2)
    # delta3_reserve = 255 - delta3
    cv2.imshow("delta3", delta3)
    cv2.waitKey(0)

def test2():
    video_path = "../SYSU/forged/00002_080-194.mp4"
    camera = cv2.VideoCapture(video_path)
    (ret1, frame1) = camera.read()
    (ret2, frame2) = camera.read()
    img = cv2.subtract(frame1, frame2)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def test3():
    p = mp.Pool(4)
    p.map(test4, range(10))


def test4(i):
    print(i)


if __name__ == "__main__":
    test3()

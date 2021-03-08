import cv2
import os

def test1():
    origin1 = cv2.imread("../images/test/prist/00002-000081.png")
    origin2 = cv2.imread("../images/test/prist/00002-000082.png")
    forged1 = cv2.imread("../images/test/forged/00002-000081.png")
    forged2 = cv2.imread("../images/test/forged/00002-000082.png")
    delta1 = cv2.subtract(origin1, origin2)
    delta2 = cv2.subtract(forged1, forged2)
    # cv2.imshow("delta1", delta1)
    # cv2.imshow("delta2", delta2)
    delta3 = cv2.subtract(delta1, delta2)
    # delta3_reserve = 255 - delta3
    cv2.imshow("delta3", delta3)
    cv2.waitKey(0)


if __name__ == "__main__":
    test1()

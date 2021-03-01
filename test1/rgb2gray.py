import cv2
import numpy as np

if __name__ == "__main__":
    img_path = "./images/000000.png"
    img = cv2.imread(img_path)
    cv2.imshow("img",img)
    print(img.shape)

    width, height = img.shape[:2][::-1]

    img_resize = cv2.resize(img, (int(width*0.5), int(height*0.5)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("img", img_resize)
    print("img_resizee shape:{}".format(np.shape(img_resize)))

    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img_gray", img_gray)
    print("img_gray shappe:{}".format(np.shape(img_gray)))
    cv2.waitKey()


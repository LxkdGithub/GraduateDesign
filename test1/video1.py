import os
import sys
import time
import cv2
import argparse

def process_video(input, output):
    if not os.path.exists(output):
        os.makedirs(output)

    vid = cv2.VideoCapture(input)
    ret, frame = vid.read()
    proc_frames = 0
    while ret:
        output_file = output + "\\{:0>6d}.png".format(proc_frames)
        cv2.imwrite(output_file, frame)
        ret, frame = vid.read()
        proc_frames += 1
    vid.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="01_02.mp4", type=str)
    parser.add_argument("--output", default="images1", type=str)
    args = parser.parse_args()
    start_time = time.time()
    process_video(args.input, args.output)

    print(
        "Method {}: Input:{}, Time taken:{}".format(sys.argv[0], args.input, time.time() - start_time)
    )

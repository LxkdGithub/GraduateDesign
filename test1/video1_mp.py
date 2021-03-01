import os
import cv2
import sys
import time
import argparse
import multiprocessing as mp


num_processes = 4
frame_jump_unit = 0


def process_video(group_number):
    vid = cv2.VideoCapture(input)
    pos_frames = int(frame_jump_unit * group_number)
    vid.set(cv2.CAP_PROP_POS_FRAMES, pos_frames)
    proc_frames = 0

    if not os.path.exists(output):
        os.makedirs(output)

    if group_number != num_processes - 1:
        not_last_seg = True
    else:
        not_last_seg = False

    ret, frame = vid.read()
    while ret:
        if not_last_seg and proc_frames == frame_jump_unit:
            break
        out_file = output + '/{:0>6d}.mbp'.format(proc_frames + pos_frames)
        cv2.imwrite(out_file, frame)
        ret, frame = vid.read()
        proc_frames += 1
    vid.release()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="01_02.mp4", type=str)
    parser.add_argument("--output", default="images1", type=str)
    args = parser.parse_args()
    input = args.input
    output = args.output

    start_time = time.time()

    num_processes = mp.cpu_count()
    vid = cv2.VideoCapture(args.input)
    frame_jump_unit = vid.get(cv2.CAP_PROP_FRAME_COUNT) // num_processes
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    vid.release()

    p = mp.Pool(num_processes)
    p.map(process_video, range(num_processes))


    print(
        "Method {}: Input:{}, Output:{}, Time taken: {}".format(
            sys.argv[0], args.input, args.output, time.time() - start_time
        )
    )

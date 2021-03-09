import os
import time
import cv2
import multiprocessing as mp


# convert prist video to images
def process_video(video_path, output):
    if not os.path.exists(video_path):
        print("Prist Video path is not exists")
        print(video_path)
        return
    if not os.path.exists(output):
        os.makedirs(output)

    prefix_path, video_name = os.path.split(video_path)
    video_id = video_name[:5]
    vid = cv2.VideoCapture(video_path)
    ret, frame = vid.read()
    proc_frames = 0
    while ret:
        output_file = output + "/" + video_id + "-{:0>6d}.png".format(proc_frames)
        cv2.imwrite(output_file, frame)
        ret, frame = vid.read()
        proc_frames += 1
    vid.release()


# extract forged images from forged video (skip the sequence which are not forged)
def extract_forge_img(video_path, output):
    if not os.path.exists(video_path):
        print("Forged Video path is not exists")
        print(video_path)
        return
    if not os.path.exists(output):
        os.makedirs(output)

    prefix_path, video_name = os.path.split(video_path)
    video_id = video_name[:5]
    idx_str = video_name[6:-4]
    print(idx_str)
    # get the index border which has been forged
    idxs = [-1, -1, -1, -1]
    idxs[0] = int(idx_str[:3])
    idxs[1] = int(idx_str[4:7])
    if len(idx_str) > 7:
        idxs[2] = int(idx_str[8:11])
        idxs[3] = int(idx_str[12:])
    vid = cv2.VideoCapture(video_path)
    ret, frame = vid.read()
    proc_frames = 0

    while ret:
        if (idxs[0] <= proc_frames <= idxs[1]) or (idxs[2] <= proc_frames <= idxs[3]):
            output_file = output + "/" + str(video_id) + "-{:0>6d}.png".format(proc_frames)
            cv2.imwrite(output_file, frame)
        ret, frame = vid.read()
        proc_frames += 1
    vid.release()


def all_split():
    prist_dir = "../SYSU/prist"
    forged_dir = "../SYSU/forged"
    output_dir = "../images"
    videos = os.listdir(prist_dir)
    videos.sort(key=lambda x: int(x[:5]))
    videos_forged = os.listdir(forged_dir)
    videos_forged.sort(key=lambda x: int(x[:5]))
    i = 0
    while i < 100:
        print("--- processing : {}".format(i))
        if i < 50:
            print("=======", videos[i], videos_forged[i])
            process_video(prist_dir + "/" + videos[i], output_dir + "/train/prist")
            extract_forge_img(forged_dir + "/" + videos_forged[i], output_dir + "/train/forged")
        else:
            print("=======", videos[i], videos_forged[i])
            process_video(prist_dir + "/" + videos[i], output_dir + "/test/prist")
            extract_forge_img(forged_dir + "/" + videos_forged[i], output_dir + "/test/forged")
        i += 1


if __name__ == "__main__":
    start_time = time.time()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input1", default="../SYSU/prist/00001.mp4", type=str)
    # parser.add_argument("--input2", default="../SYSU/forged/00001_136-262.mp4", type=str)
    # parser.add_argument("--output", default="../images", type=str)
    # args = parser.parse_args()
    # process_video(args.input1, args.output+"/train/prist")
    # extract_forge_img(args.input2, args.output+"/train/forged")

    all_split()

    print(
        "------------Time taken:{}------------".format(time.time() - start_time)
    )

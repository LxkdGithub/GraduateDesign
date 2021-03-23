import os
from tkinter import filedialog
import cv2
import net
import torch
import torch.nn.functional as F
import dataset


# 将[[0,0,0], [0,0,0]] => [0/3.0/1/2/3]
def Process(res):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_forged_list = [0] * len(res)
    for key, value in res.items():
        frameId = int(key)
        if sum(value[1]) == 0:
            pred_forged_list[frameId] = 0        # pred == 0
            if sum(value[0]) == 0:
                TN += 1
            else:
                FN += 1
        else:
            pred_forged_list[frameId] = 1        # perd == 1
            if sum(value[0]) == 0:
                FP += 1
            else:
                TP += 1
    return TP, TN, FP, FN, pred_forged_list


def fineTune(pred_forged_list):
    for i in range(len(pred_forged_list)):
        start = max(0, i - 4)
        end = min(len(pred_forged_list)-1, start + 9)
        count = 0
        for j in range(start, end+1):
            if pred_forged_list[j] == 1:
                count += 1
        if count >= 7:
            pred_forged_list[i] = 1
    return pred_forged_list


def getAccAfterTune(video_name, tuned_list):
    idx_str = video_name[6:-4]
    idxs = [-1, -1, -1, -1]
    idxs[0] = int(idx_str[:3])
    idxs[1] = int(idx_str[4:7])
    TP, TN, FP, FN = 0, 0, 0, 0
    if len(idx_str) > 7:
        idxs[2] = int(idx_str[8:11])
        idxs[3] = int(idx_str[12:])

    for i in range(len(tuned_list)):
        if tuned_list[i] == 0:
            if (idxs[0] <= i <= idxs[1]) or (idxs[2] <= i <= idxs[3]):
                FN += 1
            else:
                TN += 1
        else:
            if (idxs[0] <= i <= idxs[1]) or (idxs[2] <= i <= idxs[3]):
                TP += 1
            else:
                FP += 1
    return TP, TN, FP, FN


def splitAcrop(video_path):
    video_name = os.path.split(video_path)[1][:5]
    output = "../single_test/" + video_name
    if not os.path.exists(video_path):
        print("Target Video path is not exists")
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
        for i in range(3):
            output_file = output + "/" + "{:0>6d}-{}.png".format(proc_frames, i+1)
            cv2.imwrite(output_file, frame[:, i*280:i*280+720])
        ret, frame = vid.read()
        proc_frames += 1
    vid.release()


def getImgList(img_path, video_name):
    abs_path = os.path.abspath(img_path)
    imgs = os.listdir(img_path)
    idx_str = video_name[6:-4]
    idxs = [-1, -1, -1, -1]
    idxs[0] = int(idx_str[:3])
    idxs[1] = int(idx_str[4:7])
    with open(img_path+"/list.txt", "w") as f:
        for img in imgs:
            if img == "list.txt":
                continue
            frameId = int(img[:6])
            if (idxs[0] <= frameId <= idxs[1]) or (idxs[2] <= frameId <= idxs[3]):
                f.write(abs_path+"/"+img+" "+"1"+"\n")
            else:
                f.write(abs_path+"/"+img+" "+"0"+"\n")


def model_test(video_path,  model, device, test_loader):
    video_name = os.path.split(video_path)[1]
    img_dir_path = "../single_test/" + video_name
    model.eval()
    test_loss = 0
    correct = 0
    result = {}
    cur = 0
    total = len(test_loader.dataset)
    # total = 900
    with torch.no_grad():
        for data, target, img_crop_name in test_loader:
            # print(img_crop_name)          # '000123-1'
            if cur >= total:
                break
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data)
            loss1 = F.nll_loss(output[0], target)
            loss2 = F.nll_loss(output[1], target)
            loss3 = F.nll_loss(output[2], target)
            loss = loss1 + loss2 + 0.1 * loss3
            test_loss += loss  # sum up batch loss
            print(output[0][0:5], "\n", output[1][0:5], "\n", output[2][0:5])
            output_merge = output[0] * 0.45 + output[1] * 0.45 + output[2] * 0.1
            # print(output_merge)
            pred = output_merge.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            pred = pred.view(pred.shape[0])  # pred.shape = [batch_size]
            for i in range(len(pred)):
                imgId = img_crop_name[i][:-2]
                cropId = int(img_crop_name[i][-1]) - 1
                if imgId not in result:
                    result[imgId] = [[0, 0, 0], [0, 0, 0]]
                # 还是要确定crop_id
                result[imgId][0][cropId] = target[i].item()  # [label, res]
                result[imgId][1][cropId] = pred[i].item() # [label, res]
            print(len(result))
            del loss1, loss2, loss3, loss
            cur += 30

    test_loss /= total
    # Process result
    """ 
    1. calculate TP, TN, FP, FN
    2. get the list which is forged
    3. fine-tune the result of forged frames
    4. get the accuracy fine-tuned
    5. get the region of forged image
    """
    # 1. and 2.
    TP, TN, FP, FN, pred_forged_list = Process(result)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 0.1)
    recall = TP / (TP + FN + 0.1)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, TP + TN, total / 3,
                   300. * (TP + TN) / total))
    print('--- Accuracy: {} --  Precision: {}  --- Reacll: {}'.format(accuracy, precision, recall))

    # 3. Fine-Tune 之后再输出结果
    tuned_list = fineTune(pred_forged_list)
    print(pred_forged_list[:50])
    print(tuned_list[:50])
    # 4.
    TP, TN, FP, FN = getAccAfterTune(video_name, tuned_list)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 0.1)
    recall = TP / (TP + FN + 0.1)
    print('--- Accuracy: {} --  Precision: {}  --- Reacll: {}'.format(accuracy, precision, recall))


if __name__ == "__main__":
    test_video = "00051_137-229.mp4"
    test_video_path = os.path.abspath("../SYSU/forged/" + test_video)
    # splitAcrop(test_video_path)
    # getImgList(os.path.abspath("../single_test/" + test_video[:5]), test_video)

    model = net.Net()
    device = "cuda"
    torch.cuda.empty_cache()
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load("VideoFake_cnn-2.pt"))

    test_kwargs = {'batch_size': 30,
                   'num_workers': 3,
                   'pin_memory': True,
                   'shuffle': False}
    test_dataset = dataset.TorchDataset(
        "../single_test/" + test_video[:5] +"/list.txt",
        resize_height=720, resize_width=720,
        mode=3,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    model_test(test_video_path, model, device, test_loader)
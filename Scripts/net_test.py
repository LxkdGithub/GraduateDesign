import torch
import torch.nn.functional as F
import torch.utils.data
import net
import dataset
import os


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    result = {}
    cur = 0
    total = len(test_loader.dataset)
    # total = 900
    with torch.no_grad():
        for data, target, img_path in test_loader:
            if cur >= total:
                break
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data)
            loss1 = F.nll_loss(output[0], target)
            loss2 = F.nll_loss(output[1], target)
            loss3 = F.nll_loss(output[2], target)
            loss = loss1 + loss2 + 0.1 * loss3
            test_loss += loss  # sum up batch loss
            output_merge = output[0] * 0.5 + output[1] * 0.4 + output[2] * 0.1
            pred = output_merge.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            pred = pred.view(pred.shape[0])     # pred.shape = [batch_size]
            for i in range(len(pred)):
                if img_path[i] not in result:
                    result[img_path[i]] = [0, 0]
                result[img_path[i]][0] += target[i].item()       # [label, res]
                result[img_path[i]][1] += pred[i].item()       # [label, res]
            print(len(result))
            del loss1, loss2, loss3, loss
            cur += 30

    # 1. 计算correct
    # def isTrue(a):
    #     target = a[0]
    #     res = a[1]
    #     if (target + res == 0) or (target > 0 and res > 0):  # target 只有0/3 res 有0/1/2/3
    #         return 1
    #     return 0
    # print(len(result))
    # new_result = list(map(isTrue, result.values()))
    # correct = sum(new_result)

    # 2. 函数计算准确率 召回率
    def Accuuacy(res):
        TP = 0      # 检测为篡改 而且是真的
        TN = 0      # 检测为原始 真的
        FP = 0
        FN = 0
        for key, value in res.items():
            if value[1] == 0:
                if value[0] == 0:
                    TN += 1
                else:
                    FN += 1
            else:
                if value[0] == 0:
                    FP += 1
                else:
                    TP += 1
        return TP, TN, FP, FN

    TP, TN, FP, FN = Accuuacy(result)
    accuracy = (TP + TN) / (TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP +FN)


    print(accuracy, precision, recall)

    test_loss /= total

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, TP+TN, total/3,
        300. * (TP+TN) / total))
    print('--- Precision: {}  --- Reacll: {}'.format(accuracy, recall))


if __name__ == "__main__":
    model = net.Net()
    device = "cuda"
    torch.cuda.empty_cache()
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load("VideoFake_cnn-317-2.pt"))

    test_kwargs = {'batch_size': 30,
                   'num_workers': 3,
                   'pin_memory': True,
                   'shuffle': False}
    test_dataset = dataset.TorchDataset(
        "../images/test/Shuffle.txt",
        resize_height=720, resize_width=720,
        isTest=True,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    test(model, device, test_loader)

import torch
from torchvision import transforms, datasets, utils
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
from torch.utils.tensorboard import SummaryWriter


def Predict(model, model_weight_path):
    writer = SummaryWriter("../logs")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])

    try:
        json_file = open('class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # model = AlexNet(num_classes=10)
    # model_weight_path = "./AlexNet.pth"
    # model = BPNet()
    # model_weight_path = "./BPNet.pth"
    # model = LeNet()
    # model_weight_path = "./LeNet.pth"

    model.load_state_dict(torch.load(model_weight_path))

    img_dir_path = "../data_set/hand_write/"
    imgs = os.listdir(img_dir_path)
    origin_imgs = [img_dir_path + img for img in imgs]
    imgs = [data_transform(Image.open(img).convert("L")) for img in origin_imgs]
    imgs = [torch.unsqueeze(img, dim=0) for img in imgs]
    origin_label = [img.split("/")[-1].split("_")[0] for img in origin_imgs]
    acc = 0
    for stp in range(len(imgs)):
        writer.add_image("predict", data_transform(Image.open(origin_imgs[stp]).convert("L")), stp)

        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(imgs[stp]))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # print("测试数字为：", origin_label[stp], ", 预测数字为：", class_indict[str(predict_cla)], predict[predict_cla].item())
        acc += (origin_label[stp] + " ") == class_indict[str(predict_cla)].split("-")[0]
    print("predict acc = ", acc / 20.0)
    writer.close()

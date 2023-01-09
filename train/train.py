import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.optim as optim
from net.AlexNet import AlexNet
from net.BPNet import BPNet
import json
import time

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    writer = SummaryWriter(log_dir="../logs")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5], std=[0.5])]),

        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5], std=[0.5])])}

    train_dataset = MNIST(root='../data_set/', train=True, download=True, transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=1)

    validate_dataset = MNIST(root='../data_set/', train=False, download=True, transform=data_transform["val"])
    val_num = len(train_dataset)
    validate_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=64,
                                                  shuffle=True,
                                                  num_workers=1)

    digit_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in digit_list.items())

    json_str = json.dumps(cla_dict, indent=9)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # net = AlexNet(num_classes=10, init_weights=True)
    # save_path = 'pre_train/AlexNet.pth'
    net = BPNet()
    save_path = 'pre_train/BPNet.pth'

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    best_acc = 0.0

    for epoch in range(10):
        ########################################## train ###############################################
        net.train()
        running_loss = 0.0
        time_start = time.perf_counter()

        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()

            # writer.add_images("{0:d} epoch".format(epoch), images, step)

            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        print()
        print('%f s' % (time.perf_counter() - time_start))
        writer.add_scalar('train_time', time.perf_counter() - time_start, epoch)

        ########################################### validate ###########################################
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
                  (epoch + 1, running_loss / step, val_accurate))

            writer.add_scalar('loss', running_loss / step, epoch)
            writer.add_scalar('acc', val_accurate, epoch)

    print('Finished Training')

    writer.close()

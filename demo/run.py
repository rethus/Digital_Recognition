from train import Train
from predict import Predict
from net import AlexNet, BPNet, LeNet

if __name__ == '__main__':
    # Train(AlexNet(num_classes=10), '../pre_train/LeNet.pth', epoch_num=20)
    # Train(BPNet(), '../pre_train/BPNet.pth', epoch_num=100)
    # Train(LeNet(), '../pre_train/LeNet.pth', epoch_num=100)

    Predict(AlexNet(num_classes=10), '../pre_train/AlexNet.pth')
    Predict(BPNet(), '../pre_train/BPNet.pth')
    Predict(LeNet(), '../pre_train/LeNet.pth')
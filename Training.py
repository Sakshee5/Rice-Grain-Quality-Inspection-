import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import os
import numpy as np
import torch.nn.init
import random
from skimage import segmentation
import pandas as pd
from pre_processing import mask_and_crop

def seed_everything(seed):
    """
    To recreate segmentation results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 10
seed_everything(seed)

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=50, type=int, help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=80, type=int, help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.02, type=float, help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, help='visualization flag')
args = parser.parse_args()


# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=args.nChannel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)

        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append(nn.Conv2d(args.nChannel, args.nChannel, kernel_size=(3, 3), stride=(1, 1), padding=1))
            self.bn2.append(nn.BatchNorm2d(args.nChannel))

        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        """
        Each layer defined above has a weight and a transformation function associated with it which is defined here
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)

        x = self.conv3(x)
        x = self.bn3(x)
        return x


def unsupervised_segmentation(im):
    loss_lst = []
    im_denoise = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 0, 15)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
    data = Variable(data)

    labels = segmentation.felzenszwalb(im_denoise, scale=1, sigma=0.1, min_size=60)
    # plt.imshow(segmentation.mark_boundaries(im, labels))
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    u_labels = np.sort(u_labels)

    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

    # train
    model = MyNet(data.size(1))
    model.train()

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Set a specific seed for color mapping to ensure reproducibility
    color_seed = 42
    np.random.seed(color_seed)
    label_colours = np.random.randint(255, size=(args.nChannel, 3))

    for batch_idx in range(args.maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]  # (600, 400, 100)
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)   # (240000, 100)

        _, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        for i in range(len(l_inds)):
            labels_per_sp = im_target[l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
        target = torch.from_numpy(im_target)

        if args.visualize:
            im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

            scale_percent = 50  # percent of original size
            width = int(im_target_rgb.shape[1] * scale_percent / 100)
            height = int(im_target_rgb.shape[0] * scale_percent / 100)
            dim = (width, height)

            cv2.imshow('output', cv2.resize(im_target_rgb, dim, interpolation=cv2.INTER_AREA))
            cv2.waitKey(10)

        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        print(batch_idx, '/', args.maxIter, '|', ' label num rgb :', nLabels, 'loss :', loss.item())
        loss_lst.append(loss.item())

    df = pd.DataFrame({f"Loss Seed {seed}": loss_lst})
    df.to_excel(f"seed_{seed}_loss_values.xlsx", index=False)
    torch.save(model.state_dict(), os.path.join(r"Trained Models", 'model' + '.pth'))
    cv2.imshow('output', cv2.resize(im_target_rgb, dim, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)


img_path = r"Images/DSC01902.JPG"
img = mask_and_crop(img_path)
print("Starting Segmentation...")
unsupervised_segmentation(img)

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


def seed_everything(seed=2):
    """
    To recreate segmentation results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=50, type=int, help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=100, type=int, help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.025, type=float, help='learning rate')
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


def get_rice_mask_thru_colour_detection(img_path):
    """
    Img ---> converted to HSV color space to detect background ---> create a mask of grain ---> make sure all background
    pixels are set to black colour ---> crop grain ---> add 5 pixel black background padding on each side
    """
    img = cv2.resize(cv2.imread(img_path), (800, 1200))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([70, 110, 0])
    upper = np.array([109, 255, 255])

    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    def neighbours(im):
        for i in range(im.shape[1]):  # set top and bottom row of pixels to 0 incase they are not masked right
            im[0, i, :] = 0
            im[im.shape[0] - 1, i, :] = 0

        for i in range(im.shape[0]):  # set right and left column of pixels to 0
            im[i, 0, :] = 0
            im[i, im.shape[1] - 1, :] = 0

        neighbours = [1, 2, 3, 4, 5]
        # check top bottom right and left pixels of every pixel till it's fifth neighbour and if they are all zero then
        # set pixel in question to zero incase they were not masked right

        for val in neighbours:
            for i in range(val, im.shape[0] - val):
                for j in range(1, im.shape[1] - val):
                    if np.any(im[i, j, :]) != 0:
                        if np.all(im[i + val, j, :] == 0) and np.all(im[i, j + val, :] == 0) and np.all(
                                im[i - val, j, :] == 0) and np.all(im[i, j - val, :] == 0):
                            im[i, j, :] = 0

        return im

    def trim(frame):
        # crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        # crop bottom
        elif not np.sum(frame[-1]):
            return trim(frame[:-2])
        # crop left
        elif not np.sum(frame[:, 0]):
            return trim(frame[:, 1:])
        # crop right
        elif not np.sum(frame[:, -1]):
            return trim(frame[:, :-2])
        return frame

    final = cv2.copyMakeBorder(trim(neighbours(imgResult)), 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=0)

    return final


def unsupervised_segmentation(im):
    im_denoise = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 0, 15)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
    data = Variable(data)

    labels = segmentation.felzenszwalb(im_denoise, scale=1, sigma=0.1, min_size=60)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)

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

    torch.save(model.state_dict(), os.path.join(r"C:\Users\Sakshee\Documents\Rice Defect Dataset", 'model' + '.pth'))
    cv2.imshow('output', cv2.resize(im_target_rgb, dim, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)


img_path = r"C:\Users\Sakshee\Documents\Rice Defect Dataset\All images\DSC01902.JPG"
img = get_rice_mask_thru_colour_detection(img_path)
unsupervised_segmentation(img)

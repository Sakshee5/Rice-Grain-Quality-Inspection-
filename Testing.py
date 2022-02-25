import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import os
import numpy as np
import torch.nn.init
import random
import tqdm
import numpy as np
import cv2
from collections import defaultdict
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import statistics


parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=50, type=int, help='number of channels')
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


def test(PATH, test_img_list):
    nChannel = 50
    label_colours = np.random.randint(255, size=(nChannel, 3))

    for img_file in tqdm.tqdm(test_img_list):
        im = get_rice_mask_thru_colour_detection(os.path.join("Type 2", img_file))
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        indices = np.where(im == 0)
        coordinates = zip(indices[0], indices[1])
        data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
        data = Variable(data)
        model = MyNet(data.size(1))
        model.load_state_dict(torch.load(PATH))
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

        for coord in coordinates:
            im_target_rgb[coord[0], coord[1], :] = 0

        cv2.imwrite(os.path.join("Type 2 Model 3",  os.path.basename(img_file) + '.png'), im_target_rgb)


def get_defect_percent(image_path, good_grain):
    """
    :param image_path: path to concerned image file
    :param good_grain: RGB value of color segmented as good grain (needs to be manually extracted from by_color dict)
    :return: percent of defect (chalkiness + damage)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    by_color = defaultdict(int)
    img = Image.open(image_path)  # RGB FORMAT
    for pixel in img.getdata():
        by_color[pixel] += 1

    print(by_color)

    total_grain = image.shape[0] * image.shape[1] - by_color[(0, 0, 0)]

    del by_color[good_grain]
    del by_color[(0, 0, 0)]

    defect_grain_percent = ((sum(list(by_color.values())) - 4000) / total_grain)

    return defect_grain_percent


PATH = "model_3.pth"
test_img_list = os.listdir("Type 2")
print('Testing ' + str(len(test_img_list)) + ' images.')
test(PATH, test_img_list)


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import os
import numpy as np
import torch.nn.init
import tqdm
import numpy as np
import os
from PIL import Image
from pre_processing import mask_and_crop


parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=50, type=int, help='number of channels')
parser.add_argument('--nConv', metavar='M', default=2, type=int, help='number of convolutional layers')
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


def test(PATH, test_img_list):
    nChannel = 50
    label_colours = np.random.randint(255, size=(nChannel, 3))

    for img_file in tqdm.tqdm(test_img_list):
        im = mask_and_crop(os.path.join("Images", img_file))
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

        cv2.imwrite(os.path.join("Predictions",  os.path.basename(img_file) + '.png'), im_target_rgb)


PATH = "Trained Models/model.pth"
test_img_list = os.listdir("Images")
print('Testing ' + str(len(test_img_list)) + ' images.')
test(PATH, test_img_list)


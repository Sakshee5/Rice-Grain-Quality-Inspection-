# Rice Grain Quality Inspection
Unsupervised defect segmentation of high resolution (4000 x 6000 pixels) individual rice grain images.

The unsupervised segmentation algorithm is based on [this](https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf) research paper.

## Methodology:
### 1. Image Processing
RGB ---> HSV ---> Grain Masking with manual colour detection ---> Crop background
Since HSV colorspace is more robust to external lighting changes and thus manual mask values set for one image can be used to mask all other rice grains. To get rid of extra segmentations in the background detected due to different lighting and shadows
Background cropping is done in order to reduce computation.

### 2. CNN Forward pass
CNN is used to extract a feature representation of input image via a forward pass. Here, a (w, h, 3) input image gets represented by (w, h, c) 
feature vector where default value of c=50.

### 3. Argmax Classification of CNN output to compute target image
(w, h, c) feature vector is first flattened i.e. (w x h, c) and then subjected to argmax classification which results in a vector of shape (w x h, 1) which is inturn 
reshaped to (w, h, 1). This is considered as the target image.

### 4. Refinement of unsupervised segments/clusters using Superpixel algorithm
The target image obtained from argmax classification is further refined by clustering the segments as per the results of the SLIC (Simple Linear Iterative Clustering)
superpixel algorithm.

### 5. Loss calculation via backpropogation
Crossentropy loss is calculated between the output image (w, h, c) and the target image (w, h, 1) and minimized via back propagation by updation 
of network parameters.

**************************************************************************
Style image Lists
**************************************************************************

The style_image_lists directory contains files with the IDs of images
used for training and testing photographic style classifiers. The files are:

1. train.jpgl - list of IDs of training images
2. test.jpgl  - list of IDs of testing images
3. styles.txt - numeric style IDs and their associated photographic styles.
4. train.lab  - annotations for images in train.jpgl consisting of numeric
style IDs.
5. test.multilab - binary annotations for images in test.jpgl. There are 14 
columns corresponding to the 14 possible styles so that, for example, a 1 
in column 3 means that the image has been labeled with the 3rd style listed 
in styles.txt

Note that the training images are single-labeled, but the test images are
multilabeled.

The download link is: 
https://github.com/imfing/ava_downloader/tree/master/AVA_dataset/style_image_lists


total 11270

1 Complementary_Colors	760
2 Duotones		1041
3 HDR			317
4 Image_Grain		672
5 Light_On_White	960
6 Long_Exposure	676
7 Macro		1359
8 Motion_Blur		488
9 Negative_Image	768
10 Rule_of_Thirds	1112
11 Shallow_DOF		1184
12 Silhouettes		825
13 Soft_Focus		568
14 Vanishing_Point	540

submit: Cui Jia

# ConvNet-Search-For-Art

Dataset Overview:
"The dataset covers ten of the categories present in PASCAL VOC, and is split 
into training, validation, and test sets. Objects have a variety of sizes, poses
and depictive styles, and can be partially occluded or truncated. The paintings 
have been obtained from Your Paintings. The annotations have been provided by 
the public as part of the Tagger Project as well as having been extracted from 
the painting titles." -Department of Engineering Science, University of Oxford

In this project, we use convolutional networks to find features in paintings.

Running our code:

1. Ensure that you have the following packages installed:
keras, pillow, xlrd, numpy, requests, and BitesIO

2. Enter python Painting_ConvNet.py into the terminal.

3. Once the images are compiled, you will be asked if you would like to test
a few data points.  Enter 0 to test data points.  Enter anything else to omit
testing.

TensorFlow will begin running.  Please note it may take more than 4 hours to 
complete all epochs.

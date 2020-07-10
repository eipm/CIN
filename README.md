# Predict Chromosomal Instability Status (CIN) from Histopathology Images


[![Python 3.2](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/release/python-2715/)
[![TensorFlow 2](https://img.shields.io/badge/TF-2-orange.svg)](https://www.tensorflow.org/install/source)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


Non-invasive prediction of response to neoadjuvant chemotherapy using the preNAC algorithm. The preNAC algorithm is a method based on **GoogLeNet** [[1]](#1) architecture which is trained using MRI images obtained from [**TCIA**](https://wiki.cancerimagingarchive.net/display/Public/Breast-MRI-NACT-Pilot) [[2]](#2) [[3]](#3). 

![cin Logo](Images/cin.png)


The algorithm classify :heavy_plus_sign: & :heavy_minus_sign: response to NAC in :womens: breast cancer using one of the A:eye: mathods.

To run the preNAC framework please follow these steps:

:one: Install the TensorFlow. Follow the instruction from here: https://www.tensorflow.org/install/

:writing_hand: We use TensorFlow V1 in this project.

:two: Pre-trained Models of CNN architectures should be downloaded from the "Pre-trained Models" part of https://github.com/wenwei202/terngrad/tree/master/slim#pre-trained-models and be located in your machine (e.g. Breast_MRI_NAC/script/slim/run/checkpoint). The files for pre-trained models (e.g. inception_v1.ckpt) are available under the column named "checkpoint".

:three: Divide the images with the original size into two or more classes based on the aim of classification (e.g., classification of BR-P-NAC and BR-N-NAC). 85% of images in each class will be selected as Train set (train and validation) and 15% for Test set. 

:four: _NUM_CLASSES should be set in MRI.py (this script is located in Breast_MRI_NAC/script/slim/datasets).

:five: Run the convert.py (it is located in the "Breast_MRI_NAC/script" directory) to allocate the suitable percentage of images to train and validation sets. 

```bash
python convert.py ../Images/train process/ 0
```

The convert.py needs three arguments including: 

![#ffdce0](https://via.placeholder.com/10/ffdce0/000000?text=+) `../Images/train` :arrow_right: `the address of images for training`

![#ffdce0](https://via.placeholder.com/10/ffdce0/000000?text=+) `process/` :arrow_right: `the address of where the result will be located`

![#ffdce0](https://via.placeholder.com/10/ffdce0/000000?text=+) `0` :arrow_right: `the percentage of validation images for the training step `

:writing_hand: Keep the percentage of validation images as 0 because we set 15% for validation inside the code.

:writing_hand: It will save converted .tf records in the "process" directory, so make sure the "process" folder is empty before running.


:six: The Inception-V1 architecture should be run on the train set images from the "Breast_MRI_NAC/scripts/slim" directory. First go to the the "Breast_MRI_NAC/scripts/slim" directory. You can change the parameters (e.g. batch_size, optimizer, and learning_rate) of the "load_inception_v1.sh" file that is located in "run/" directory. Then, run the following command in shell script: 

```bash
./run/load_inception_v1.sh
```

:writing_hand: If you got the bash error like permission denied, run the following line in your shell:

```bash
chmod 777 load_inception_v1.sh
```

:writing_hand: Each script in slim dataset should be run separately based on the selected architecture. The slim folder contains some sub-folders. 

:writing_hand: You can set up the parameters of each architectures in the run sub-folder. For example you can set the architecture in a way to run from scratch or trained for the last or all layer. Also you can set the batch size or the number of maximum steps. 

:writing_hand: You can see the "result" folder at "script/result" as the result of running the above script. So, make sure the "result" folder is empty before running.

:seven: The trained algorithms should be tested using test set images. In folder "Breast_MRI_NAC/script/slim", predict.py loads a trained model on provided images. This code get 5 argu/resultments:

```bash
python predict.py v1 ../result/ ../../Images/test output.txt 2
```


![#ffdce0](https://via.placeholder.com/10/ffdce0/000000?text=+) `v1` :arrow_right: `inception-v1`

![#ffdce0](https://via.placeholder.com/10/ffdce0/000000?text=+) `../result/` :arrow_right: `the address of trained model`

![#ffdce0](https://via.placeholder.com/10/ffdce0/000000?text=+) `../Images/test` :arrow_right: `the address of test set images`

![#ffdce0](https://via.placeholder.com/10/ffdce0/000000?text=+) `output.txt` :arrow_right: `the output result file`

![#ffdce0](https://via.placeholder.com/10/ffdce0/000000?text=+) `2` :arrow_right: `number of classes`



## References
<a id="1">[1]</a> 
Szegedy, C. et al. (2015). 
Going deeper with convolutions.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 
DOI :10.1109/cvpr.2015.7298594.

<a id="2">[2]</a> 
Clark, K. et al. (2013). 
The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository.
Journal of Digital Imaging. 
DOI :10.1007/s10278-013-9622-7.

<a id="3">[3]</a> 
Newitt, D. et al. (2016). 
The Single site breast DCE-MRI data and segmentations from patients undergoing neoadjuvant chemotherapy.
The Cancer Imaging Archive. 
DOI :10.7937/K9/TCIA.2016.QHsyhJKy.

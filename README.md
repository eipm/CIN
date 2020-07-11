## Predict Chromosomal Instability Status (CIN) from Histopathology Images


[![Python 3.2](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/release/python-2715/)
[![TensorFlow 2](https://img.shields.io/badge/TF-2-orange.svg)](https://www.tensorflow.org/install/source)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


The CIN algorithm is a method based on **Densenet** [[1]](#1) architecture which is pre-trained using ImageNet images [[2]](#2).

![cin Logo](Image/cin.png)


The algorithm classify genome CIN+/CIN- based on H&E histology whole slide images (WSI)

To run the CIN framework please follow these steps:

:one:   **Software:** 

Install the following softwares used in this project:

*TensorFlow2*: Follow the instruction from here: https://www.tensorflow.org/install/

*OpenSlide*： Python version. See more details here: https://openslide.org/

:two:   **Image Preprocessing**

See codes under the folder '/Codes/Preprocessing'

1. Tile WSI

Use the function of **tiling_wsi** to get the best WSI from raw svs file.

The default setting will get WSI on 2.5x magnification with dimension of 2048x2048. Through setting *tilesize* and *overlap* can change the fine grade of sliding windowns.

You need to define two *list* objects ahead:  *filepaths* and *samplenames*. Also set the argument of *tile_dir* for the location to store WSI.

`filepaths : list of svs file paths`

`samplenames : list of sample names with the same order of filepaths. For example: 'TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A'`

`tile_dir : location to store WSI. For example: '/Image/WSI/'`


2. Crop WSI
:three:   **Feature Extraction**

:four:  

:five: 

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
Huang, G. et al. (2017). 
Densely Connected Convolutional Networks.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 
DOI :10.1109/CVPR.2017.243.

<a id="2">[2]</a> 
@inproceedings{imagenet_cvpr09,
        AUTHOR = {Deng, J. and Dong, W. and Socher, R. and Li, L.-J. and Li, K. and Fei-Fei, L.},
        TITLE = {{ImageNet: A Large-Scale Hierarchical Image Database}},
        BOOKTITLE = {CVPR09},
        YEAR = {2009},
        BIBSOURCE = "http://www.image-net.org/papers/imagenet_cvpr09.bib"}

<a id="3">[3]</a> 
Newitt, D. et al. (2016). 
The Single site breast DCE-MRI data and segmentations from patients undergoing neoadjuvant chemotherapy.
The Cancer Imaging Archive. 
DOI :10.7937/K9/TCIA.2016.QHsyhJKy.

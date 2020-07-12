## Predict Chromosomal Instability Status (CIN) from Histopathology Images


[![Python 3.2](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/release/python-2715/)
[![TensorFlow 2](https://img.shields.io/badge/TF-2-orange.svg)](https://www.tensorflow.org/install/source)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


The CIN algorithm is a method based on **Densenet** [[1]](#1) architecture which is pre-trained using ImageNet images [[2]](#2).

![cin Logo](Image/cin.png)


The algorithm classifies genome CIN+/CIN- based on H&E histology whole slide images (WSI)

To run the CIN framework please follow these steps:

###   **Software** 

Install the following softwares used in this project:

*TensorFlow2*: Follow the instruction from here: https://www.tensorflow.org/install/

*OpenSlide*： Python version. See more details here: https://openslide.org/

###   **Genome CIN Score** 

See codes for genome CIN score calculation at '/Codes/Genome CIN/CIN_SCORE_CALCULATION.R'

###   **Image Preprocessing**

See codes under the folder '/Codes/Preprocessing'

#### Tile WSI

![tilewsi](Image/tiling_wsi.png)

Use the function of **tiling_wsi** to get the best WSI from raw svs file. The default setting will get WSI on 2.5x magnification with dimension of 2048x2048. Through setting *tilesize* and *overlap* can change the step size of sliding windows.

You need to define two *list* objects ahead:  *filepaths* and *samplenames*. Also set the argument of *tile_dir* for the location to store WSI.

`filepaths : list of svs file paths`

`samplenames : list of sample names with the same order of filepaths. For example: 'TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A'`

`tile_dir : location to store WSI. For example: '/Image/WSI/'`


#### Crop WSI

![cropwsi](Image/WSIcropping.png)

Use the function of **WSIcropping** to crop input images into 8x8 nonoverlapping grids and save patches into target location. You can set up a tissue percentage threshold for QC. This function outputs patches and organize them in the structure of one folder per patient. Please refer to '/Image/Patch/'

`inputdir : input directory. example: /Image/WSI/`

`targetdir : target directory. example: /Image/Patch/`

`pct : tissue percentage threshold. Default is 80%`


###   **Feature Extraction**

![featureextract](Image/feature_extract.png)

See codes at '/Codes/Feature Extraction/FeatureExtraction.py'

First, set lab reference for color normalization. 

```bash
ref_img_path='/Image/WSI/IMAGE_NAME.jpg'
```

Use the function of **Densenet121_extractor** to extract patient level features from cropped patches of last step stored in '/Image/Patch'. You need to organize a pandas DataFrame (WSI_df) to store the patient information including patch location and CIN score that will be used as labels after. A sample dataframe can be found in '/File/WSI_df.csv'. This function will automatically conduct color normalization. 

`WSI_df : a pandas dataframe with all patients' patches will be extracted, should contain column of Barcode_Path with all the paths of patients' folder. example: /Image/Patch/TCGA-3C-AALJ`

`target : target path of patient level features to be stored. example: '/Bottlenect_Features/features_densenet121.npy'`


###   **MLP Training**




## References
<a id="1">[1]</a> 
Huang, G. et al. (2017). 
Densely Connected Convolutional Networks.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 
DOI :10.1109/CVPR.2017.243.

<a id="2">[2]</a> 
Deng, J. et al. (2009).
ImageNet: A Large-Scale Hierarchical Image Database.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
http://www.image-net.org/papers/imagenet_cvpr09.bib

# EFPN
WSI Cancer Tissue Classification
﻿
This repository provides a Python implementation of a multi-scale framework for classifying cancer tissue in Whole Slide Images (WSIs) using Multi-Instance Learning (MIL).
﻿
Overview
﻿
The framework is designed to assist in the automated classification of cancerous and normal tissues from WSIs, a task that is crucial for improving the efficiency and accuracy of pathology diagnostics.
﻿
Key Components
﻿
- Preprocessing Module: Selects informative patches from WSIs.
- Efficient Feature Pyramid Network (EFPN): Extracts multi-scale features from the selected patches.
- Similarity Focal Loss: Custom loss function to enhance model training on challenging samples.
﻿
Getting Started
﻿
To use the code, follow these steps:
﻿
﻿
1. Install Dependencies
Ensure you have Python and PyTorch installed. You may also need to install additional libraries specified in `requirements.txt`.
2.Download the Dataset
Once you have downloaded the dataset, modify the `train.py` script by setting the `--train-data-path` and `--val-data-path` to the absolute path of the extracted dataset folder.
3.Download Pre-trained Weights
Obtain the pre-trained weights required for training.
4.Set Pre-trained Weights Path
In the `train.py` script, set the `--weights` parameter to the path where you have saved the pre-trained weights.
5.Start Training
With the dataset path `--data-path` and the pre-trained weights path `--weights` correctly set, you can now start training using the `train.py` script. During the training process, a `class_indices.json` file will be automatically generated.
6.Use Your Own Dataset
If you are using your own dataset, please arrange it according to the classification structure (i.e., one category corresponds to one folder, and within this, another folder for each package). Also, set the `num_classes` in both the training and prediction scripts to match the number of categories in your dataset.

# Final-Project-Group8

## Installation

**First, install the Chrome browser driver:**

macOS

`brew install chromedriver`

**Second, create and activate a Python virtual environment:** 

`python3 -m venv testenv
source testenv/bin/activate
pip3 install -r requirements.txt`

**To view the presentation, install Jupyter and RISE:**

[Jupyter](https://jupyter.org/install)

[RISE](https://rise.readthedocs.io/en/maint-5.6/installation.html)

## Data
All of the data for this project can be found on [Google Drive](https://drive.google.com/drive/folders/14YJngXIdbD-_D3qks1_uSd5Pnh2vfCad?usp=sharing). The data utilized comes from the [Kaggle](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018) competition [iMaterialist Challenge (Fashion) at FGVC5](https://github.com/visipedia/imat_fashion_comp).

## Code
1) Clone this repository. Then, download the Data and Models directories from [Google Drive](https://drive.google.com/drive/folders/14YJngXIdbD-_D3qks1_uSd5Pnh2vfCad?usp=sharing).

2) Download data:

***Disclaimer - it takes ~6 hours to download the fashion dataset***

- download_fashion_dataset.py - used to download the fashion images from fashion image urls
- obtain_fashion_dataset_labels.py - used to create the train, validation, and test csv files that will be used in the data loader for modeling

3) Exploratory data analysis:

- helper_functions.py - used to find the smallest image size and check the balance of the labels
- presentation_visualizations.py - used to create various visualizations used in the presentation

4) Train models:

***Disclaimer - it takes ~4-12 hours to train each of these models on a NVIDIA Tesla P100 GPU on Google Cloud Platform***

- baseline_model_template.py - used as a template to create the simple CNNs
- jessica_model_1-7.py & mary_model_1-2.py - used to train and save the simple CNNs
- mobilenet_model.py - used to train and save the mobilenetv2 model

5) Test models: 

densenet_model.py - used to test the densenet161 model
test_models.py - used to test the simple CNNs and mobilenetv2 model

6) Recommendations:

- scrape_banana_republic_images.py - used to scrape women's fashion images from the Banana Republic website
- baseline_recommendations.py - used to generate baseline KNN ranking recommendations
- generate_recommendations.py - used to obtain CNN feature extraction-based KNN ranking recommendations

## Summary
#### Problem
We all have our favorite pieces of clothing that we consistently wear. Over time, these clothing items may not fit anymore or degrade in quality. Subsequently, it may be difficult to find the same clothing item or time-intensive to find similar clothing items due to the vast amount of clothing websites and clothing retail stores. This is a common issue for both of us, and it is something we would like to streamline!
#### Solution
We built a deep learning-based fashion recommendation system. Our solution involves a two-step approach. Our solution involves a two-step approach. First, we train a convolutional neural network on fashion images in order to extract the feature maps associated with different clothing items. Second, we use the feature maps as input to a KNN model that will find the five closest neighbors to a given query image that will serve as recommendations.
#### Recommendations
![Jeans Recommendations](https://github.com/mgibbs1259/Final-Project-Group8/blob/master/Final-Group-Presentation/jeans_recommendations.png)

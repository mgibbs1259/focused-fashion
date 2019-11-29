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

## Summary
#### Problem
We all have our favorite pieces of clothing that we consistently wear. Over time, these clothing items may not fit anymore or degrade in quality. Subsequently, it may be difficult to find the same clothing item or time-intensive to find similar clothing items due to the vast amount of clothing websites and clothing retail stores. This is a common issue for both of us, and it is something we would like to streamline!
#### Solution
We built a deep learning-based fashion recommendation system. Our solution involves a two-step approach. Our solution involves a two-step approach. First, we train a convolutional neural network on fashion images in order to extract the feature maps associated with different clothing items. Second, we use the feature maps as input to a KNN model that will find the five closest neighbors to a given query image that will serve as recommendations.
#### Recommendations
![Jeans Recommendations](https://github.com/mgibbs1259/Final-Project-Group8/blob/master/Final-Group-Presentation/final-group-presentation/jeans_recommendations.png)
<br>
![Skirt Recommendations](https://github.com/mgibbs1259/Final-Project-Group8/blob/master/Final-Group-Presentation/final-group-presentation/skirt_recommendations.png)
<br>
![Jessica Recommendations](https://github.com/mgibbs1259/Final-Project-Group8/blob/master/Final-Group-Presentation/final-group-presentation/jessica_recommendations.png)

## References
https://github.com/visipedia/imat_fashion_comp
https://www.kaggle.com/nlecoy/imaterialist-downloader-util

https://www.kaggle.com/anqitu/for-starter-json-to-multilabel-in-24-seconds
https://stackoverflow.com/questions/52855843/multi-label-classification-in-pytorch
https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/
https://github.com/utkuozbulak/pytorch-custom-dataset-examples
https://www.kaggle.com/mratsim/starting-kit-for-pytorch-deep-learning

https://homes.cs.washington.edu/~bboots/files/GuerinBMVC18.pdf
https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

https://www.kaggle.com/renatobmlr/pytorch-densenet-as-feature-extractor
https://towardsdatascience.com/simple-implementation-of-densely-connected-convolutional-networks-in-pytorch-3846978f2f36

https://github.com/spotify/annoy
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree

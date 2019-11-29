import json
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

data_path = "/home/ubuntu/Final-Project-Group8/Final-Project-Group8/Code/"

train={}
test={}
validation={}
with open('%s/train.json'%(data_path)) as json_data:
    train= json.load(json_data)
with open('%s/test.json'%(data_path)) as json_data:
    test= json.load(json_data)
with open('%s/validation.json'%(data_path)) as json_data:
    validation = json.load(json_data)

print('Train No. of images: %d'%(len(train['images'])))
print('Test No. of images: %d'%(len(test['images'])))
print('Validation No. of images: %d'%(len(validation['images'])))

# Train
train_img_url=train['images']
train_img_url=pd.DataFrame(train_img_url)
train_ann=train['annotations']
train_ann=pd.DataFrame(train_ann)
train=pd.merge(train_img_url, train_ann, on='imageId', how='inner')

# Test
test=pd.DataFrame(test['images'])

# Validation
val_img_url=validation['images']
val_img_url=pd.DataFrame(val_img_url)
val_ann=validation['annotations']
val_ann=pd.DataFrame(val_ann)
validation=pd.merge(val_img_url, val_ann, on='imageId', how='inner')

# Create test with labels
test2 = train_ann[700000:730000]


# Only keep first 500000 rows
train_ann = train_ann[:500000]

drop = pd.read_csv('/home/ubuntu/Final-Project-Group8/Final-Project-Group8/drop_2.csv')

drop = drop['imageId'].tolist()

train_ann['imageId'] = train_ann['imageId'].astype(int)

train_ann = train_ann[~train_ann['imageId'].isin(drop)]


datas = {'Train': train, 'Test': test, 'Validation': validation}
for data in datas.values():
    data['imageId'] = data['imageId'].astype(np.uint32)

# Write out val_ann and train_ann as csv
# val_ann.to_csv("validation.csv")
# test2.to_csv("test.csv")

# Convert labels using the multilabelbinarizer
mlb = MultiLabelBinarizer()
train_label = mlb.fit_transform(train['labelId'])
validation_label = mlb.transform(validation['labelId'])
test_label = mlb.transform(test2['labelId'])
dummy_label_col = list(mlb.classes_)
print(dummy_label_col)

for data in [validation_label, train_label, test_label]:
    print(data.shape)

# Dataframes of the labels
train_label = pd.DataFrame(data=train_label, columns=list(mlb.classes_))
train_label.head()
# train_label.to_csv("train_label")
validation_label = pd.DataFrame(data=validation_label, columns=list(mlb.classes_))
validation_label.head()
# train_label.to_csv("validation_label")
test_label = pd.DataFrame(data=validation_label, columns=list(mlb.classes_))
test_label.head()
# train_label.to_csv("test_label")

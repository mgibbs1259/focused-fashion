import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

validation = pd.read_csv("validation.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

mlb = MultiLabelBinarizer()
train_label = mlb.fit_transform(train['labelId'])
validation_label = mlb.transform(validation['labelId'])
test_label = mlb.transform(test['labelId'])
dummy_label_col = list(mlb.classes_)
print(dummy_label_col)

for data in [validation_label, train_label, test_label]:
    print(data.shape)

# Save as numpy
dummy_label_col = pd.DataFrame(columns = dummy_label_col)
# dummy_label_col.to_csv('%s/dummy_label_col.csv'%'', index = False)
# np.save('%s/dummy_label_train.npy' % '', train_label)
# np.save('%s/dummy_label_val.npy' % '', validation_label)
dummy_label_col.head()

# Save as csv if you prefer
# train_label = pd.DataFrame(data = train_label, columns = list(mlb.classes_))
# train_label.head()
# validation_label = pd.DataFrame(data = validation_label, columns = list(mlb.classes_))
# validation_label.head()

#Works best when run right after Load_labels.py

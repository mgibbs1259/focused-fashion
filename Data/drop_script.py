import os
import pandas as pd

TRAIN_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_train"

drop = pd.read_csv('drop_these.csv')
drop_files = drop['bad'].tolist()

# for file in drop_files:
#     file_path = os.path.join(TRAIN_DIR, file)
#     os.remove(file_path)

print(len(os.listdir(TRAIN_DIR)))

import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import MultiLabelBinarizer


#Load Data
df_txt = pd.read_csv('new_model_number_6.txt', header = None)
df1 = pd.read_csv("train_ann_drop.csv")
label_map = pd.read_csv("label_map.csv")


def clean_txt_file(txt_file, new_txt_file):
    """Use this to clean up the .txt files for use
    Removes lines containing strings we wish to remove and outputs to new txt file.
    Load the new txt file to plot the data"""
    words_to_remove = ['MODEL_NAME', 'EPOCH', 'Validation']
    with open(txt_file) as oldfile, open(new_txt_file, 'w') as newfile:
        for line in oldfile:
            if not any(words_to_remove in line for words_to_remove in words_to_remove):
                newfile.write(line)


def plot_loss(df):
    """Plots the loss for each Batch"""
    df = df.rename(columns={0: "Batch number", 1: "Loss"})
    loss = df[['x','Loss']] = df['Loss'].str.split(':',expand=True)
    df = df.drop(['x'], axis=1)
    Batch_number = df[['x','Batch number']] = df['Batch number'].str.split(':',expand=True)
    df = df.drop(['x'], axis=1)
    df["Loss"] = pd.to_numeric(df["Loss"])
    ax = df.plot(lw = 1, colormap = 'jet',x='Batch number', y=['Loss'], figsize=(20,10), grid=True, title='Train Loss')
    ax.set_xlabel("Batch number")
    ax.set_ylabel("Loss")
    fig = ax.get_figure()
    fig.savefig("loss_plot.png")


def explore_labels(df):
    """Cleans data, creates histogram, gets descriptive stats of the number of
    per image"""
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop(['Unnamed: 0.1'], axis=1)
    df['Length'] = df.labelId.apply(lambda x: len(x))
    df['labelId'] = df['labelId'].apply(literal_eval)
    x = df['Length'].describe()
    ax = df['Length'].hist(bins=25)
    fig = ax.get_figure()
    fig.savefig("label_histogram.png")
    return x


def prep_data_for_cloud(df, label_map):
    """Prepared data for cloud visualization"""
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop(['Unnamed: 0.1'], axis=1)
    df['labelId'] = df['labelId'].apply(literal_eval)
    mlb = MultiLabelBinarizer()
    Labels = mlb.fit_transform(df['labelId'])
    Labels = pd.DataFrame(data=Labels, columns=list(mlb.classes_))
    # Obtain count of each label
    Label_Count = Labels.sum()
    df2 = Label_Count.to_frame().reset_index()
    df2 = df2.rename(columns={0: "count", 'index': "Label"})
    label_map = label_map.rename(columns={"labelId": "Label"})
    df2["Label"] = df2["Label"].astype(int)
    # Merge the dataframes together
    mergeddf = df2.merge(label_map, on='Label')
    mergeddf = mergeddf.drop(['taskId'], axis=1)
    mergeddf = mergeddf.drop(['taskName'], axis=1)
    mergeddf = mergeddf.drop(['Label'], axis=1)
    return mergeddf


def create_cloud(mergeddf):
    """Create word cloud visualization"""
    #Create frequency dictionary
    d = {}
    for word, count in mergeddf.values:
        d[count] = word
    #Create word cloud visualization
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies = d)
    plt.figure()
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis = 'off'
    plt.savefig("word_cloud")
    plt.show()


clean_txt_file("model_number_6.txt", "new_model_number_6.txt")
plot_loss(df_txt)
explore_labels(df1)
mergeddf = prep_data_for_cloud(df1, label_map)
create_cloud(mergeddf)

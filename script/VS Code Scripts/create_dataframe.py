import pandas as pd
import numpy as np

# import the cuurent state of dataframes
df1 = pd.read_csv("../../data/instagram_data/captions_csv.csv", low_memory=False)
df2 = pd.read_csv("../../data/instagram_data2/captions_csv2.csv", low_memory=False)


# making changes in dataframe1
df1 = df1.drop(columns=['Sr No'])
df1["Image File"] = "../../data/instagram_data/" + df1["Image File"] + ".jpg"


# making changes in dataframe2
df2 = df2.drop(df2.columns[0], axis=1)

first_row = pd.DataFrame([df2.columns.tolist()])
first_row.columns = ["Image File", "Caption"]

df2.columns = first_row.columns
df2 = pd.concat([first_row, df2], ignore_index=True)

df2["Image File"] = "../../data/instagram_data2/" + df2["Image File"] + ".jpg"


# Merging both df1 and df2
df = pd.concat([df1, df2], ignore_index=True)

df = df.dropna(subset=["Caption"])

# Adding the Folds column to train against cross validation
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df['Fold'] = np.tile(np.arange(5), int(np.ceil(len(df) / 5)))[:len(df)]

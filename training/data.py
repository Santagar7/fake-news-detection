import pandas as pd
from sklearn.model_selection import train_test_split


def download_and_prepare_data():
    Real = pd.read_csv("true.csv")
    Real['Class'] = 'Real'
    Fake = pd.read_csv("fake.csv")
    Fake['Class'] = 'Fake'
    Fake = Fake[:2000]
    Real = Real[:2000]
    df = pd.concat([Real, Fake], ignore_index=True)
    df = df.dropna()
    class_to_int = {"Fake": 0, "Real": 1}
    df['Class'] = df['Class'].replace(class_to_int)
    return df


def split_data(df):
    df_train, df_test = train_test_split(df, test_size=0.30, shuffle=True)
    df_val, df_test = train_test_split(df_test, test_size=0.50, shuffle=True)
    return df_train, df_val, df_test

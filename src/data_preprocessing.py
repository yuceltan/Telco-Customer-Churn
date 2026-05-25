import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_STATE, TEST_SIZE


def load_data(file_path):

    df = pd.read_csv(r'C:\Users\Yuceltan Ebiri\PycharmProjects\Telco-Customer-Churn\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df


def clean_data(df):


    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna(subset=["TotalCharges"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def split_features_target(df):

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def get_column_types(X):

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numerical_features, categorical_features


def build_preprocessor(X):


    numerical_features, categorical_features = get_column_types(X)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", encoder, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numerical_features, categorical_features


def create_train_test_split(X, y):


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path: str):
    df = pd.read_csv(path, encoding='ISO-8859-1')  # or encoding='cp1252'
    return df


def feature_engineering(df: pd.DataFrame):
    # Example features — modify as per your dataset
    df['Total_Spend'] = df['Quantity'] * df['UnitPrice']
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': 'max',
        'Total_Spend': 'sum',
        'InvoiceNo': 'count'
    }).reset_index()

    rfm.rename(columns={'InvoiceDate':'LastPurchase',
                        'Total_Spend':'Monetary',
                        'InvoiceNo':'Frequency'}, inplace=True)
    return rfm

def scale_features(df: pd.DataFrame):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.select_dtypes(include=['float64','int64']))
    return scaled

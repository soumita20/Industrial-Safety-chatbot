from sklearn.preprocessing import LabelEncoder
import pandas as pd

def label_encode(data):
    lb_make = LabelEncoder()
    data['Accident Level'] = lb_make.fit_transform(data['Accident Level'])
    data['Potential Accident Level'] = lb_make.fit_transform(
        data['Potential Accident Level'])
    return data

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)

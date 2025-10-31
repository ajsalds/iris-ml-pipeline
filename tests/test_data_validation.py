import pandas as pd
from src.data_validation import validate_data_df

def test_validate_data_ok():
    df = pd.DataFrame({
        'sepal_length': [5.1],
        'sepal_width': [3.5],
        'petal_length': [1.4],
        'petal_width': [0.2],
        'species': ['setosa']
    })
    assert validate_data_df(df)

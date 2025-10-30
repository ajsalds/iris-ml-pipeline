import pandas as pd

def validate_data_df(df: pd.DataFrame) -> bool:
    required_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
    # check columns
    if not required_columns.issubset(df.columns):
        return False
    # basic sanity: no nulls
    if df[list(required_columns)].isnull().any().any():
        return False
    return True

if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1])
    ok = validate_data_df(df)
    print("VALID" if ok else "INVALID")
    sys.exit(0 if ok else 1)

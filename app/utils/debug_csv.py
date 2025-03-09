import pandas as pd

if __name__ == "__main__":
    # Update the CSV path as needed
    recipes_csv = "C:/Users/fluk2/Desktop/SE/Year3-2/Information_Retrieval/Project/resource/recipes.csv"

    # Read only the first 100 rows
    df = pd.read_csv(recipes_csv, low_memory=False, nrows=20)

    # Set the row index you want to display
    row_index = 1  # Change this to display another row

    # Get the selected row as a dictionary
    row_data = df.iloc[row_index].to_dict()

    # Print in "column-name: value" format
    for key, value in row_data.items():
        print(f"{key}: {value}")

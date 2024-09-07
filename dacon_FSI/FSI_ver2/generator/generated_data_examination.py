import pandas as pd

# Load the dataset
data = pd.read_csv("/workspace/Dataset/FSI/generated_data.csv")
# data = pd.read_csv("/workspace/Personal_Development/dacon_fraud/classifier/run/train6/labels_predictions.csv")
# Print column names, number of rows, and number of columns
print("Columns:", data.columns.tolist())
print("Total number of Rows:", len(data))
print("Total number of Columns:", len(data.columns))

# Iterate over the columns and print their names
for i in range(len(data.columns)):
    print(f"{i+1}th column: {data.columns[i]}")

if len(data.columns) >= 63:
    column_63_values_count = data.iloc[:, 62].value_counts()
    total_distinct_values = len(column_63_values_count)
    print("Value counts for the 63rd column:") 
    print(column_63_values_count)
    print("Total number of distinct values in the 62rd column:", total_distinct_values)

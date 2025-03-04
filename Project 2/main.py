import pandas as pd

# Load the dataset
file_path = "/Users/maximcrucirescu/Desktop/dtu notes/sem 6/math_modelling/githubvenv/Project 2/Libian_desert_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
df.head()

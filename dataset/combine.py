import pandas as pd
import os

# Path to your dataset folder
dataset_folder = "C:/Users/kdhas/Desktop/Research New/Dataset/new dataset"

# Collect all CSV filenames
csv_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]

# Load and combine
combined_df = pd.concat([
    pd.read_csv(os.path.join(dataset_folder, file)) for file in csv_files
], ignore_index=True)

# Save combined dataset
output_file = os.path.join(dataset_folder, "combined_dataset.csv")
combined_df.to_csv(output_file, index=False)

print(f"\n Combined dataset saved to: {output_file}")
print(f"Total records: {len(combined_df)}")
print(f"Categories: {combined_df['type'].unique()}")
print("\n Sample rows:\n")
print(combined_df.head(20))


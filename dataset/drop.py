import pandas as pd
import os

# === Configuration ===
input_file = 'C:/Users/kdhas/Desktop/Research New/Dataset/5G_Traffic_Dataset/Zoom_2.csv' 
column_to_drop = 'No.'   
output_folder = 'C:/Users/kdhas/Desktop/Research New/Dataset/drop_dataset'
output_file = os.path.join(output_folder, 'new_Zoom_2.csv')

# === Load CSV ===
df = pd.read_csv(input_file)

# === Drop Column ===
if column_to_drop in df.columns:
    df.drop(columns=[column_to_drop], inplace=True)
else:
    print(f"Column '{column_to_drop}' not found in the CSV.")

# === Create Output Folder if it doesn't exist ===
os.makedirs(output_folder, exist_ok=True)

# === Save Modified CSV ===
df.to_csv(output_file, index=False)
print(f"Modified CSV saved to: {output_file}")

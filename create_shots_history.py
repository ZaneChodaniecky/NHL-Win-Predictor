# -*- coding: utf-8 -*-


import requests
import zipfile
import os
import pandas as pd
import shutil
from io import BytesIO

# List of zip file URLs
zip_urls = [
    "https://peter-tanner.com/moneypuck/downloads/shots_2015.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2016.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2017.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2018.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2019.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2020.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2021.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2022.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2023.zip"
]

# Function to download and extract a zip file
def download_and_extract_zip(url, extract_to="data_files"):
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Files extracted to {extract_to}")
    else:
        print(f"Failed to download {url}")

# Function to read all CSV files in a folder and combine them
def combine_csv_files(folder_path):
    # List all files in the directory
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    # Read each CSV file into a pandas DataFrame
    data_frames = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
        print(f"Loaded {file}")
    
    # Combine all DataFrames into one
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    return combined_data

# Function to delete a folder and its contents
def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder {folder_path}")

# Download and extract each zip file, and combine their CSV data
combined_data = pd.DataFrame()  # Start with an empty DataFrame

for i, zip_url in enumerate(zip_urls, start=1):
    # Extract files to a sub-folder for each zip
    data_folder = f"data_files_{i}"
    download_and_extract_zip(zip_url, data_folder)
    
    # Combine CSV files from the extracted folder
    data = combine_csv_files(data_folder)
    
    # Append this data to the final combined data
    combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    # Delete the extracted folder after processing
    delete_folder(data_folder)

# Save the combined data to a new CSV file
combined_data.to_csv(r"Data/shots_history.csv", index=False)

print("All data combined successfully and saved as 'shots_history.csv'!")

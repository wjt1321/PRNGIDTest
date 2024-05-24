# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:48:55 2024

@author: jarre
"""

# I used chatgpt to help me write this code, I asked it to help me write a script
# that would combine multiple csv's into one using pandas. 
import pandas as pd
import glob
import os



def combine_csv_files(input_folder, output_file):
    # Specify the path or pattern for your CSV files
    csv_files = glob.glob(f'{input_folder}/*.csv')

    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()

    # Iterate through the list of CSV files
    for file in csv_files:
        # Read each CSV file into a DataFrame
        df = pd.read_csv(file, header=None)

        # Concatenate the current DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True, axis=0)

    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    homedirectory = os.getcwd()

    # Specify the input folder containing CSV files and the output file
    input_folder = homedirectory + "/results"
    output_file =  homedirectory + '/output.csv'

    # Call the function to combine CSV files
    combine_csv_files(input_folder, output_file)

    print(f'Combined CSV file saved to {output_file}')

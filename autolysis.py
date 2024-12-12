
# Required files are given in meta data as requires and dependencies. So no need to install each time in pip command
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn",
#   "requests",
#   "openai",
#   "tabulate",
#   "chardet",
#   "python-dotenv"
# ]
# ///

import os
import sys
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import chardet
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Setup AI Proxy token
AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

#Detect the encoding of the csv file
def encoding_file(file_path): 
  with open(file_path, 'rb') as f: 
    result = chardet.detect(f.read()) 
    return result['encoding'] 

# Load the CSV file
def load_data(file_path): 
  try: 
    encoding = encoding_file(file_path) 
    df = pd.read_csv(file_path, encoding=encoding) 
    print(f"Successfully read {file_path} with encoding {encoding}")
    return df 
  except Exception as e:
    print(f"Failed to read {file_path}: {e}")
    raise

# Create some histograms using the data and combine them into a single image.
def combine_histograms(df, columns, output_file):
    num_cols = 3  # Number of columns in the grid
    num_rows = (len(columns) + num_cols - 1) // num_cols  # Compute the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 5))

    # Flatten axes if it's a numpy.ndarray 
    if isinstance(axes, np.ndarray): 
       axes = axes.flatten()

    for i, col in enumerate(columns):
        # ax = axes[i // num_cols, i % num_cols]
        ax=axes[i]
        sns.histplot(df[col], kde=True, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    # Remove any empty subplots
    for j in range(i + 1, num_rows * num_cols):
        # fig.delaxes(axes[j // num_cols, j % num_cols])
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Create correlation matrix using the numerical cols. Also, call the combine_histograms function.
def generate_visualizations(df, output_dir, agg_threshold=10):
    os.makedirs(output_dir, exist_ok=True)

    # Set the font
    plt.rcParams['font.family'] = 'Arial'

    # Filter out the types of columns for easier visualizations
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    datetime_columns = df.select_dtypes(include=['datetime64']).columns

    # Combines histograms into one PNG
    combine_histograms(df, numeric_columns, os.path.join(output_dir, 'combined_histograms.png'))

    # Correlation matrix heatmap
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()

# Perform Generic Analysis such as the summary, missing values and correlation matrix.
def generic_analysis(df):
    summary_stats = df.describe(include='all')
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=['number']) 
    correlation_matrix = numeric_df.corr()
    return summary_stats,missing_values,correlation_matrix

# Detects outliers in the data
def detect_outliers(df):
   imputer = SimpleImputer(strategy='mean') 
   imputed_df = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include='number')), columns=df.select_dtypes(include='number').columns) 
   iso_forest = IsolationForest(contamination=0.1, random_state=42) 
   outliers = iso_forest.fit_predict(imputed_df) 
   return outliers



# Creating the prompt by using the stats we found earlier
def generate_prompt(summary_stats, missing_values, correlation_matrix):
    prompt = f"""
    I have a dataset with the following summary statistics:
    {summary_stats}

    Missing values:
    {missing_values}

    Correlation matrix:
    {correlation_matrix}

    Identify key insights and write a story about this analysis.
    """
    return prompt
   
# The Most Important Part! Using our favourite LLM to give us a story
def ask_llm(prompt):

    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {
        "model": "gpt-4o-mini",  # The latest model
        "messages": [{"role": "user", "content": prompt}],
        "temperature":0.7, # 0.7 is for a creative output. can be lowered if creativity isn't necessary!
        "max_tokens":1000
    }
    
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=payload) 
    
    if response.status_code != 200: 
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}") 
    response_json = response.json()
    #print(response_json) # printing it to know some information on it!
    message_content=response_json['choices'][0]['message']['content'] # We needed only this!
    
    return message_content

#Using all the above function, this function provides the output that was aimed
def process_dataset(dataset_name,dataset_path):
   df=load_data(dataset_path)

   generate_visualizations(df,dataset_name)

   summary_statments,missing_values,correlation_matrix=generic_analysis(df)

   df['Outliers']=detect_outliers(df)


   
   prompt=generate_prompt(summary_statments,missing_values,correlation_matrix)
   story = ask_llm(prompt)
   
   # Creating README.md for each dataset with the Summary and Story

    # Save results in a Markdown README

   with open(os.path.join(dataset_name, "README.md"), "w") as f:
        f.write(story)
        print("Readme saved")


     

# Running if the file is run as the main file
if __name__=='__main__':
    if len(sys.argv) != 2:
        raise ValueError("Usage: uv run autolysis.py dataset.csv")
    dataset_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    process_dataset(dataset_name,dataset_path)

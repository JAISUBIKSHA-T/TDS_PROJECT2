# Required files are given in meta data as requires and dependencies. So no need to install each time in pip command
# /// script
# requires-python = ">=3.11"
# requires-openai=">=0.27.0"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scikit-learn",
#   "chardet",
#   "openai",
#   "statsmodels",
#   "networkx",
#   "geopandas",
#   "scipy"
# ]
# ///


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import httpx
import chardet
import time
import base64
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Set the AIPROXY TOKEN from environment variable
api_key = os.getenv("AIPROXY_TOKEN")
AIPROXY_TOKEN = api_key

# Set the API URL for querying the LLM
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# Load the CSV file with automatic encoding detection
def load_data(file_path):
    """
    Load CSV data while automatically detecting the encoding to prevent read errors.
    
    Args:
        file_path (str): Path to the CSV file to load.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # Detect the file's encoding
        encoding = result['encoding']
        data = pd.read_csv(file_path, encoding=encoding)  # Load CSV with the detected encoding
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

# Perform basic analysis on the dataset
def basic_analysis(data):
    """
    Generate basic statistical analysis and summarize missing values and column types.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
    
    Returns:
        dict: A dictionary with summary statistics, missing values, and column types.
    """
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    column_info = data.dtypes.to_dict()
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Perform outlier detection using Interquartile Range (IQR)
def outlier_detection(data):
    """
    Detect outliers in the dataset using the IQR method.
    
    Args:
        data (pd.DataFrame): The dataset to analyze.
    
    Returns:
        dict: A dictionary containing the number of outliers per numeric column.
    """
    numeric_data = data.select_dtypes(include=np.number)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}

# Combine histograms for specified columns
def combine_histograms(data, columns, output_dir):
    """
    Create histograms for specified columns and save them as a single image.
    
    Args:
        data (pd.DataFrame): The dataset to visualize.
        columns (list): List of column names to plot histograms for.
        output_dir (str): Directory to save the generated histogram image.
    
    Returns:
        str: Path to the saved histogram image.
    """
    num_cols = 3
    num_rows = (len(columns) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 5))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(data[col], kde=True, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    histogram_path = os.path.join(output_dir, "histogram.png")
    plt.savefig(histogram_path)
    plt.close()

    return histogram_path

# Perform DBSCAN clustering
def dbscan_clustering(data, output_dir):
    """
    Apply DBSCAN clustering algorithm and generate a scatter plot of clusters.
    
    Args:
        data (pd.DataFrame): The dataset to cluster.
        output_dir (str): Directory to save the generated cluster plot.
    
    Returns:
        str: Path to the saved cluster plot.
    """
    numeric_data = data.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    numeric_data['cluster'] = clusters

    x_col = numeric_data.columns[0]
    y_col = numeric_data.columns[1]

    plt.figure(figsize=(8, 6))
    scatterplot = sns.scatterplot(
        x=numeric_data.iloc[:, 0],
        y=numeric_data.iloc[:, 1],
        hue=numeric_data['cluster'],
        palette="viridis",
        legend="full"
    )
    scatterplot.set_xlabel(x_col)
    scatterplot.set_ylabel(y_col)
    scatterplot.set_title("DBSCAN Clustering")
    scatterplot.legend(title="Cluster")

    dbscan_path = os.path.join(output_dir, "dbscan_clusters.png")
    plt.savefig(dbscan_path)
    plt.close()

    return dbscan_path

# Generate a correlation heatmap
def generate_visualizations(data, output_dir):
    """
    Generate visualizations including heatmaps for correlation analysis.
    
    Args:
        data (pd.DataFrame): The dataset to visualize.
        output_dir (str): Directory to save the visualizations.
    
    Returns:
        str: Path to the saved heatmap.
    """
    numeric_data = data.select_dtypes(include=np.number).dropna()
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
    heatmap_path = f"{output_dir}/heatmap.png"
    plt.title("Correlation Heatmap")
    plt.savefig(heatmap_path)
    plt.close()
    return heatmap_path

# Query the LLM for analysis or generation of content
def query_llm_for_analysis(prompt):
    """
    Send a prompt to the LLM API and handle retries for rate limiting.
    
    Args:
        prompt (str): The prompt to send to the LLM.
    
    Returns:
        str: The response from the LLM.
    """
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    retries = 10
    backoff_factor = 2
    max_wait_time = 60

    for attempt in range(1, retries + 1):
        try:
            response = httpx.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = min(backoff_factor ** attempt, max_wait_time)
                print(f"Rate limit hit (attempt {attempt}/{retries}), retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error: {e.response.status_code}, message: {e.response.text}")
                break
        except httpx.RequestError as e:
            print(f"Request error: {e}")
            break

    print("Max retries reached, giving up.")
    sys.exit(1)

# Generate a markdown readme file with the analysis results
def save_readme(content, output_dir):
    """
    Save the generated narrative content to a markdown file.
    
    Args:
        content (str): The content to write to the README file.
        output_dir (str): The directory where the README file should be saved.
    """
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(content)
    print("Readme saved")

# Main function to analyze the dataset and generate output
def analyze_and_generate_output(file_path):
    """
    Analyze the dataset, perform clustering, generate visualizations, and create a report.
    
    Args:
        file_path (str): Path to the dataset (CSV file).
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_data(file_path)
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    analysis = basic_analysis(data)
    outliers = outlier_detection(data)
    combined_analysis = {**analysis, **outliers}
    
    image_paths = {
        'combine_histogram': combine_histograms(data, numeric_columns, output_dir),
        'dbscan_clusters': dbscan_clustering(data, output_dir),
        'heatmap': generate_visualizations(data, output_dir),
    }
    
    formatted_image_paths = "\n".join(
        f"- {key.replace('_', ' ').title()}: {path}"
        for key, path in image_paths.items()
    )

    data_info = {
        "filename": file_path,
        "summary": combined_analysis["summary"],
        "missing_values": combined_analysis["missing_values"],
        "outliers": combined_analysis["outliers"]
    }
    
    prompt = (
        "You are a creative storyteller tasked with creating a narrative based on a dataset analysis. "
        "Please structure the narrative as follows:\n\n"
        "1. **Introduction**:\n"
        "   - Provide an overview of the dataset, its purpose, and key insights from the data summary.\n\n"
        "2. **Data Quality Assessment**:\n"
        "   - Highlight missing values and how they might impact the analysis.\n\n"
        "3. **Outlier Analysis**:\n"
        "   - Discuss the outliers detected, their potential implications, and how they are addressed.\n\n"
        "4. **Visual Insights**:\n"
        "   - Describe the visualizations provided and what they reveal about the dataset.\n\n"
        "5. **Clustering Analysis**:\n"
        "   - Explain the clustering results (e.g., DBSCAN) and any patterns observed among the clusters.\n\n"
        "Here is the analysis information for reference:\n\n"
        f"**Data Summary**:\n{data_info['summary']}\n\n"
        f"**Missing Values**:\n{data_info['missing_values']}\n\n"
        f"**Outlier Analysis**:\n{data_info['outliers']}\n\n"
        f"**Visualizations**:\n{formatted_image_paths}\n\n"
        "Based on this, craft a compelling narrative covering all the above points in order."
    )

    narrative = query_llm_for_analysis(prompt)
    save_readme(narrative, output_dir)

# Main function execution
def main():
    """
    Main entry point of the program. Expects a CSV file as input and processes it.
    """
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()







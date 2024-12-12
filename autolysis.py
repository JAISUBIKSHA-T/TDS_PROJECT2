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






# Set the AIPROXY TOKEN
api_key = os.getenv("AIPROXY_TOKEN")

AIPROXY_TOKEN= api_key

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# Load the 'csv' file
def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # Detect encoding
        encoding = result['encoding']
        data = pd.read_csv(file_path, encoding=encoding)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

# Perform basic analysis like summary stats, missing values, etc.
def basic_analysis(data):
    summary = data.describe(include='all').to_dict()  # Summary statistics
    missing_values = data.isnull().sum().to_dict()  # Missing values
    column_info = data.dtypes.to_dict()  # Column types
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Robust outlier detection using IQR (Interquartile Range)
def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}


def combine_histograms(data, columns, output_dir):
    num_cols = 3  # Number of columns in the grid
    num_rows = (len(columns) + num_cols - 1) // num_cols  # Compute the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 5))

    # Flatten axes if it's a numpy.ndarray 
    if isinstance(axes, np.ndarray): 
       axes = axes.flatten()

    for i, col in enumerate(columns):
        # ax = axes[i // num_cols, i % num_cols]
        ax=axes[i]
        sns.histplot(data[col], kde=True, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    # Remove any empty subplots
    for j in range(i + 1, num_rows * num_cols):
        # fig.delaxes(axes[j // num_cols, j % num_cols])
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    histogram_path = os.path.join(output_dir, "histogram.png")
    plt.savefig(histogram_path)
    
    plt.close()

    return histogram_path

    
# DBSCAN clustering (Density-Based Clustering)
def dbscan_clustering(data, output_dir):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    numeric_data['cluster'] = clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=numeric_data['cluster'], palette="viridis")
    plt.title("DBSCAN Clustering")
    dbscan_path = os.path.join(output_dir, "dbscan_clusters.png")
    plt.savefig(dbscan_path)
    print("dbscan_clusters.png created")
    plt.close()
    return dbscan_path


  
def gen_stat_visual(data, output_dir):
    
    # Filter numeric columns and drop NaN values
    data1 = data.select_dtypes(include=np.number).dropna()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate the filename dynamically based on visualization_type
    
    output_path = os.path.join(output_dir, "heatmap.png")
    prompt1 = "You are to generate a Python code for the given task. Only output the code and nothing else."   
    prompt2 = (
            
            f"""Generate Python code to create a heatmap plot using the seaborn library and save it as a PNG file in the directory '{output_dir}'.
The dataset is a Pandas DataFrame named `data`. Use all numeric columns for the heatmap.Give graph title according to the data. 
The filename for saving the plot should be '{output_path}'. Return only executable Python code."""  )
        

    # API request payload
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            
            {"role": "system", "content": prompt1},
            {"role": "user", "content": prompt2},
        ],
        "max_tokens": 500,
        "temperature": 0.7,
    }

    # API call
    response = httpx.post(API_URL, headers=headers, json=payload)
    response_data = response.json()

    if "choices" in response_data and response_data["choices"]:
        code = response_data["choices"][0]["message"]["content"]
        
        code = code.replace("```python", "").replace("```", "").strip()

        print("Cleaned generated code:\n", code)

        # Validate the code before execution (basic check for malicious content)
        forbidden_keywords = ["exec(", "eval(", "__import__", "open("]
        if any(keyword in code for keyword in forbidden_keywords):
            raise ValueError("Generated code contains unsafe operations.")

        # Save the code to a file for debugging (optional)
        with open(os.path.join(output_dir, "generated_code.py"), "w") as file:
            file.write(code)


        

        # Execute the code in a controlled environment
        try:
            local_vars = {"data": data1, "output_dir": output_dir}
            exec(code, globals(), local_vars)
        except Exception as e:
            print("Error during code execution:")
            print(code)
            raise e
    else:
        raise ValueError("Failed to get code from API response.")

     # Verify the file was created
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Graph file not found at {output_path}. Check the generated code or output directory.")

    return output_path




# Function to send the data info to the LLM and request analysis or code
def query_llm_for_analysis(prompt):
    
    

    # Prepare the prompt to query the LLM
    
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {
        "model": "gpt-4o-mini",  # or use the correct model
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    retries = 10  # Increased retries before giving up
    backoff_factor = 2  # Exponential backoff factor
    max_wait_time = 60  # Maximum wait time (1 minute) to prevent indefinite retries

    for attempt in range(retries):
        try:
            response = httpx.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Exponential backoff with a cap on wait time
                wait_time = min(backoff_factor ** attempt, max_wait_time)
                print(f"Rate limit hit, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error querying the LLM: {e}")
                break
        except httpx.RequestError as e:
            print(f"Error querying the LLM: {e}")
            break

    print("Max retries reached, giving up.")
    sys.exit(1)  # Exit after retries have failed

# Save results in a Markdown README
def save_readme(content, output_dir):
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(content)
        print("Readme saved")

# Function to analyze and generate output for each file
def analyze_and_generate_output(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_data(file_path)
    numeric_columns = data.select_dtypes(include=['number']).columns
    print("Data loaded")
    
    # Perform basic analysis
    analysis = basic_analysis(data)
    outliers = outlier_detection(data)
    combined_analysis = {**analysis, **outliers}

    # Generate visualizations and save file paths
    image_paths = {}
    image_paths['combine_histogram']=combine_histograms(data, numeric_columns, output_dir)
    image_paths['correlation_matrix'] = generate_correlation_matrix(data, output_dir)
    image_paths['dbscan_clusters'] = dbscan_clustering(data, output_dir)
    image_paths['heatmap'] = gen_stat_visual(data, output_dir)
    
    print("Images created:\n", image_paths)

    # Format image paths for the prompt
    formatted_image_paths = "\n".join(
        f"- {key.replace('_', ' ').title()}: {path}"
        for key, path in image_paths.items()
    )
    
    # Send data to LLM for analysis and suggestions
    data_info = {
        "filename": file_path,
        "summary": combined_analysis["summary"],
        "missing_values": combined_analysis["missing_values"],
        "outliers": combined_analysis["outliers"]
    }
    
    prompt = (
        "You are a creative storyteller. "
        "Craft a compelling narrative based on this dataset analysis:\n\n"
        f"Data Summary: {data_info['summary']}\n\n"
        f"Missing Values: {data_info['missing_values']}\n\n"
        f"Outlier Analysis: {data_info['outliers']}\n\n"
        f"Visualizations:\n{formatted_image_paths}\n\n"
        "Create a narrative covering these points."
    )

    narrative = query_llm_for_analysis(prompt)
    print(f"\nLLM Narrative:\n{narrative}")

    # Save the narrative to a README file
    save_readme(narrative, output_dir)



# Main execution function
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()

# Required files are given in meta data as requires and dependencies. So no need to install each time in pip command

# /// script
# requires-python = ">=3.11"
# requires-openai=">=0.27.0"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "requests",
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
import requests
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image

# Set the AIPROXY TOKEN
#api_key = os.getenv("AIPROXY_TOKEN")
AIPROXY_TOKEN='eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDE2OTZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.RI1VedMQmvVJGVO63TULkf-w86U0U7kWg_qd9baBxMU'

api_key=AIPROXY_TOKEN

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# Load the 'csv' file
def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # Detect the encoding of a file that may have different or unknown character encodings from different sources or systems.
        encoding = result['encoding']   # which encoding among these('utf-8', 'ISO-8859-1','Windows-1252')
        data = pd.read_csv(file_path, encoding=encoding)   # prevent issues where characters from different languages or symbols may appear as garbage or unreadable text when reading the CSV file
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

 #Robust outlier detection using IQR (Interquartile Range)        #Outliers - to identify extreme values in your dataset that could be errors or just rare events.

def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1              
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}

# Correlation Matrix

def generate_correlation_matrix(data, output_dir):   # To find relationships between multiple variables
    data = data.select_dtypes(include=[np.number])
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.xlabel(data.columns[0], fontsize=12)
    plt.ylabel(data.columns[1], fontsize=12)
    plt.title("Correlation Matrix")
    corr_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(corr_path)
    plt.close()
    return corr_path

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
    plt.xlabel(numeric_data.columns[0], fontsize=12)
    plt.ylabel(numeric_data.columns[1], fontsize=12)    
    plt.title("DBSCAN Clustering")
    dbscan_path = os.path.join(output_dir, "dbscan_clusters.png")
    plt.savefig(dbscan_path)
    print("dbscan_clusters.png created")
    plt.close()
    return dbscan_path


def gen_stat_visual(data, output_dir, visualization_type):
    
    # Filter numeric columns and drop NaN values
    data1 = data.select_dtypes(include=np.number).dropna()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate the filename dynamically based on visualization_type
    sanitized_type = visualization_type.replace(" ", "_").lower()  # Sanitize filename
    output_path = os.path.join(output_dir, f"{sanitized_type}.png")
    

    # Prompts for the API
    prompt1 = "You are to generate a Python code for the given task. Only output the code and nothing else."
    if visualization_type=="heat map" :
        prompt2 = (
        f"""Generate Python code to create a {visualization_type} plot using the seaborn library and save it as a PNG file and filename as {output_path} in the directory '{output_dir}'.
        The code is run in an interperter so do not add the \"python\" command in the front.
        The dataset is a Pandas DataFrame named `data`. Use all numeric columns for the heatmap.Give graph title according to the data. 
        
        Return only executable Python code."""
    )
    else:    
        prompt2 = (
        f"""Generate Python code to create a {visualization_type} plot using the seaborn library and save it as a PNG file and filename as {output_path} in the directory '{output_dir}'.
        filename={output_path}.
        The code is run in an interperter so do not add the \"python\" command in the front.
        The dataset is a Pandas DataFrame named `data`. X values for the plot should be `data1.columns[0]` and Y values should be `data1.columns[1]`.
        Give proper names for X axis, Y axis and chart title.
        Return only executable Python code."""
    )

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
    response = requests.post(API_URL, headers=headers, json=payload)
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

    


# Function to convert image to Base64  
def image_to_base64(image_path, save_path):   #ensure the integrity of binary data during transmission
    with Image.open(image_path) as img:
        target_width = 800
        width, height = img.size
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)  # Maintain aspect ratio

        # Resize the image using LANCZOS filter
        resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)    # compress image before send to prompt

        # Ensure the directory exists before saving the image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print("Save path created")

        # Save the resized image to the given file path
        resized_img.save(save_path, format="PNG")   # Save the image as PNG (or adjust format if needed)
        print("Saved")

        # Save the resized image to a BytesIO object (in-memory binary stream)
        img_byte_arr = BytesIO()
        resized_img.save(img_byte_arr, format="PNG")
        
        img_byte_arr.seek(0)  # Go to the start of the BytesIO buffer

        # Encode the binary data to Base64
        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        
    return img_base64  # Return Base64-encoded image

# Function to send the data info to the LLM and request analysis or code
def query_llm_for_analysis(prompt):
    
    

    # Prepare the prompt to query the LLM
    
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {
        "model": "gpt-4o-mini", 
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
    print("Data loaded")
    
    # Perform basic analysis
    
    analysis = basic_analysis(data)
    outliers = outlier_detection(data)
    combined_analysis = {**analysis, **outliers}

    
    # Generate visualizations and save file paths
    image_paths = {}
    image_paths['correlation_matrix'] = generate_correlation_matrix(data, output_dir)
    image_paths['dbscan_clusters'] = dbscan_clustering(data, output_dir)
    image_paths['regression map']=gen_stat_visual(data, output_dir, 'regression map')
    image_paths['heat map']=gen_stat_visual(data, output_dir, 'heat map')
    print("Images created:\n", image_paths)
    
    
    images_base64, filenames = process_images(image_paths, output_dir)

    # Example output to verify
    print("Base64 Encoded Images (Keys Only for Verification):")
    print("keys:",list(images_base64.keys()))  # To show the keys, without the lengthy Base64 data

    print("Resized Image Filenames for LLM Analysis:")
    print("filenames after resize:",filenames)  # Send these filenames to LLM for analysis

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
        "Create a narrative covering these points:\n"
        f"Correlation matrix:{filenames[0]},\n"
        f"DBSCAN Clusters: {filenames[1]},\n"
        f"Regression map: {filenames[2]}\n"
        f"Heat map: {filenames[3]}\n"
        
            )
    narrative = query_llm_for_analysis(prompt)
    print(f"\nLLM Narrative:\n{narrative}")

    # Save the narrative to a README file
    save_readme(narrative, output_dir)

def resize_image(input_path, output_path, size=(300, 300)):
    """Resize an image to the specified size."""
    with Image.open(input_path) as img:
        img = img.resize(size)
        img.save(output_path)
    print(f"Image saved to {output_path}")

def image_to_base64(image_path):
    """Convert an image to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_images(image_paths, output_dir, resize_size=(300, 300)):
    """Resize images, convert to Base64, and store filenames for LLM analysis."""
    images_base64 = {}
    filenames = []

    for description, path in image_paths.items():
        # Step 1: Resize the image
        resized_image_path = os.path.join(output_dir, f"resized_{description}.png")
        resize_image(path, resized_image_path, size=resize_size)
        
        # Step 2: Convert the resized image to Base64
        base64_data = image_to_base64(resized_image_path)
        images_base64[description] = base64_data
        
        # Step 3: Collect the filename (without the lengthy Base64 data)
        filenames.append(resized_image_path)
    
    return images_base64, filenames


# Main execution function
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()

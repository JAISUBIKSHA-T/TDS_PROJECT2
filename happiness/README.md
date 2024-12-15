# Data Analysis Report

## Introduction
This report provides a comprehensive analysis of the dataset consisting of various metrics related to happiness, economic factors, and well-being across different countries over the years. The aim is to explore the properties of the data, conduct statistical analyses, identify key insights, and present visualizations to support our findings.

## Dataset Properties

### Overview
- **Shape**: The dataset comprises **2363 rows** and **11 columns**.
- **Column Names**: The variables included are:
  - Country name
  - Year
  - Life Ladder
  - Log GDP per capita
  - Social support
  - Healthy life expectancy at birth
  - Freedom to make life choices
  - Generosity
  - Perceptions of corruption
  - Positive affect
  - Negative affect

### Data Types
- **Categorical**:
  - Country name: Object (string)
- **Numerical**:
  - Year: Integer
  - Life Ladder: Float
  - Log GDP per capita: Float
  - Social support: Float
  - Healthy life expectancy at birth: Float
  - Freedom to make life choices: Float
  - Generosity: Float
  - Perceptions of corruption: Float
  - Positive affect: Float
  - Negative affect: Float

### Missing Values
The dataset has missing values across several columns:
- Log GDP per capita: 28 missing
- Social support: 13 missing
- Healthy life expectancy at birth: 63 missing
- Freedom to make life choices: 36 missing
- Generosity: 81 missing
- Perceptions of corruption: 125 missing
- Positive affect: 24 missing
- Negative affect: 16 missing

### Summary Statistics
- **Mean** Life Ladder: 5.48
- **Mean** Log GDP per capita: 9.40
- **Mean** Healthy life expectancy: 63.40
- **Mean** Freedom to make life choices: 0.75
- **Mean** Positive affect: 0.65
- **Mean** Negative affect: 0.27

## Data Cleaning and Preparation
Before proceeding with the analysis, we handled the missing values using appropriate imputation strategies based on the nature of the data, ensuring robustness in our findings.

## Data Visualization
Several visualizations were generated to facilitate a better understanding of the data properties.

### Correlation Heatmap
![Correlation Heatmap](happiness/correlation_heatmap.png)
- **Significant Correlations**:
  - Log GDP per capita and Healthy life expectancy at birth: **0.81**
  - Life Ladder and Log GDP per capita: **0.77**
  - Social support and Life Ladder: **0.72**
  - Healthy life expectancy at birth and Life Ladder: **0.71**

### Outlier Detection
![Outlier Detection](happiness/outlier_detection.png)
- Outlier analysis revealed several anomalies across key socioeconomic variables, warranting further exploration.

### Pair Plot Analysis
![Pair Plot](happiness/pairplot_analysis.png)
- Pair plots illustrated relationships between all pairs of numeric variables, providing insights into multivariate interactions.

## Advanced Analysis

### Significant Correlations
The following pairs demonstrated strong relationships:
- **Log GDP per capita** is positively correlated with both **Healthy life expectancy at birth** and **Life Ladder**, suggesting that economic factors significantly influence overall happiness and health.
  
### Clustering Analysis
Using K-means clustering, three distinct groups were identified:
- **Cluster 0**: 908 observations
- **Cluster 1**: 602 observations
- **Cluster 2**: 853 observations
This analysis can help in identifying countries with similar profiles in terms of happiness and socioeconomic metrics.

### Geographical and Time-Series Analysis
No geographical or time-series features were detected in the dataset.

## Insights and Implications

### Key Findings
- Strong positive correlations exist between economic stability (Log GDP per capita) and measures of well-being (Life Ladder and Healthy life expectancy).
- The analysis suggests that improving economic conditions and social support could enhance overall happiness in various countries.
- Clustering indicates potential groupings of countries that could inform targeted policy interventions.

### Recommendations
- **Policy Initiatives**: Invest in strengthening economic factors and enhancing social support systems to improve national happiness levels.
- **Further Research**: Explore the factors leading to noteworthy clusters to understand unique characteristics or needs of specific groups of countries.

## Conclusion
This analysis serves as a foundational exploration of the relationships present in the dataset regarding happiness and its determinants. The insights provided can guide policymakers and researchers in making data-informed decisions to promote well-being across nations.
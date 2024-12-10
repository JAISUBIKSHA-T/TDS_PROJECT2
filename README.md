#                                                      PROJECT2 
# *Description*
This project is to read any types of csv files and generate graphs according to the data and analyse the visualizations using llm. This project is checked against 3 csv files.
3 folders are created in the names of the csv files and the graphs related to them are stored in their respective folders.

# *Highlights*

* Required files are given in meta data as requires and dependencies. So no need to install each time in pip command.
* Detect the encoding of a file that may have different or unknown character encodings from different sources or systems using Chardet.
* Encode the file which encoding among these('utf-8', 'ISO-8859-1','Windows-1252')
* Summary, missing values,correlation matrix, dbscan clustering, Quartiles, Hierarchical clustering are found and these images are sent to llm to genreate anaysis
* Before that the images are compressed and convert into base 64( to prevent data corruption and ensure the integrity of binary data during transmission)
* Data is sent to LLM to analyze and made visualizations of the data and narrate a story about the data
* Narrative is stored in Readme.md
* The code file autolysis.py isevaluated through evaluate.py and the results are stored in results.csv
    

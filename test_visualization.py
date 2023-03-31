from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_extra.cluster import KMedoids
import pandas as pd
import numpy as np
import json
from openpyxl import Workbook

# Sample data
with open('./kmedoids/3_filtered.json') as f:
    data = json.load(f)

# Convert data to DataFrame
df = pd.DataFrame(data)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(df['abstrak'])

# Cluster the documents using K-Medoids algorithm
k = 5
kmedoids = KMedoids(n_clusters=k).fit(X)

# Assign the cluster labels to each document
labels = kmedoids.labels_

# Add the cluster labels to the DataFrame
df['cluster'] = labels

# Create a new Excel workbook and add a worksheet
workbook = Workbook()
worksheet = workbook.active

# Add the column headers to the worksheet
worksheet.append(['Title', 'Cluster'])
df_sorted = df.sort_values('cluster')
# Add the data to the worksheet
for index, row in df_sorted.iterrows():
    worksheet.append([row['judul'], row['cluster']])

# Save the workbook to a file
workbook.save('clustering_results.xlsx')
import os
import wget
import zipfile
import numpy as np
import pandas as pd
from collections import Counter

url = 'http://snap.stanford.edu/ogb/data/nodeproppred/products.zip'
output_dir = '/data/'
current_path = os.getcwd() 

# Download and unzip products folder if not already exists
if not os.path.exists(current_path + '/products.zip'):
    filename = wget.download(url)
    with zipfile.ZipFile(current_path + '/products.zip', 'r') as zip_ref:
        zip_ref.extractall(current_path)

# Create data folder if not already exists
if not os.path.exists(current_path + output_dir):
    os.makedirs(current_path + output_dir)

# Extract feature vectors, labels, training, validation & test nodes and write them to binary files
feature = pd.read_csv(current_path + '/products/raw/node-feat.csv.gz', compression='gzip', header = None).values
feature.astype(np.float32).tofile(current_path + output_dir + 'feat.bin')

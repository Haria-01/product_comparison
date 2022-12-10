#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
from scipy.spatial.distance import cosine
import pandas as pd

nlp = spacy.load("en_core_web_lg")

# Read in the datasets
dataset1 = pd.read_csv("amz_com-ecommerce_sample.csv", encoding="unicode_escape")
dataset2 = pd.read_csv("flipkart_com-ecommerce_sample.csv", encoding="unicode_escape")

# Get the product names from the datasets
product_names1 = dataset1["product_name"]
product_names2 = dataset2["product_name"]

# Prompt the user for a product name
product_name = input("Enter a product name: ")

# Get the spaCy document for the user-specified product name
product_doc = nlp(product_name)

# Initialize a list to store the cosine similarities
cosine_similarities = []

# Iterate over the product names from the first dataset
for name in product_names1:
    # Get the spaCy document for the current product name
    name_doc = nlp(name)
    
    # Convert the documents to numerical vectors
    product_vec = product_doc.vector
    name_vec = name_doc.vector
    
    # Compute the cosine similarity between the vectors
    sim = 1 - cosine(product_vec, name_vec)
    
    # Store the cosine similarity in the list
    cosine_similarities.append(sim)

# Find the product name from the first dataset with the highest cosine similarity
most_similar_index = cosine_similarities.index(max(cosine_similarities))
most_similar_product = product_names1[most_similar_index]

# Create a DataFrame to store the results
results = pd.DataFrame(columns=["Product name in Flipkart", "Retail Price in Flipkart", "Discounted Price in Flipkart",
                                "Product name in Amazon", "Retail Price in Amazon", "Discounted Price in Amazon"])

# Add the most similar product from the first dataset to the DataFrame
results = results.append({"Product name in Amazon": most_similar_product, 
                         "Retail Price in Amazon": dataset1["retail_price"][most_similar_index],
                         "Discounted Price in Amazon": dataset1["discounted_price"][most_similar_index],
                         "Product name in Flipkart": "N/A",
                         "Retail Price in Flipkart": "N/A",
                         "Discounted Price in Flipkart": "N/A"}, 
                        ignore_index=True)

# Repeat the process for the product names from the second dataset
# Initialize a list to store the cosine similarities
cosine_similarities = []

# Iterate over the product names from the first dataset
for name in product_names2:
    # Get the spaCy document for the current product name
    name_doc = nlp(name)
    
    # Convert the documents to numerical vectors
    product_vec = product_doc.vector
    name_vec = name_doc.vector
    
    # Compute the cosine similarity between the vectors
    sim = 1 - cosine(product_vec, name_vec)
    
    # Store the cosine similarity in the list
    cosine_similarities.append(sim)

# Find the product name from the first dataset with the highest cosine similarity
most_similar_index = cosine_similarities.index(max(cosine_similarities))
most_similar_product = product_names1[most_similar_index]

# Add the most similar product from the first dataset to the DataFrame
results = results.append({"Product name in Amazon": 'N/A', 
                         "Retail Price in Amazon": 'N/A',
                         "Discounted Price in Amazon": 'N/A',
                         "Product name in Flipkart": most_similar_product,
                         "Retail Price in Flipkart": dataset2["retail_price"][most_similar_index],
                         "Discounted Price in Flipkart": dataset2["discounted_price"][most_similar_index]}, 
                        ignore_index=True)

for column in results.columns:
    mode=results[column][1]
    if results[column][0]=='N/A':
      results[column][0]=results[column][1]

results.head(1)


# In[11]:


# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the two data sets
df1 = pd.read_csv('amz_com-ecommerce_sample.csv',encoding='unicode_escape')
df2 = pd.read_csv('flipkart_com-ecommerce_sample.csv',encoding='unicode_escape')

# Ask the user to input a product name
product_name = input('Enter a product name: ')

# Extract the product names, retail prices, and discount prices from the data sets
products1 = df1['product_name'].tolist()
products2 = df2['product_name'].tolist()
prices1 = df1['retail_price'].tolist()
prices2 = df2['discounted_price'].tolist()
discounts1 = df1['discounted_price'].tolist()
discounts2 = df2['discounted_price'].tolist()

# Use the TfidfVectorizer to convert the product names into numerical vectors
vectorizer = TfidfVectorizer()
vectors1 = vectorizer.fit_transform(products1)
vectors2 = vectorizer.transform([product_name])

# Find the index of the product in data set 1 with the highest cosine similarity to the user-specified product
index1 = cosine_similarity(vectors1, vectors2).argmax()

# Find the index of the product in data set 2 with the highest cosine similarity to the user-specified product
index2 = cosine_similarity(vectors2, vectors1).argmax()

# Print the results in a tabular column format
print('Product name\tRetail price\tDiscount price')
print(products1[index1], '\t', prices1[index1], '\t', discounts1[index1])
print(products2[index2], '\t', prices2[index2], '\t', discounts2[index2])


# In[6]:


get_ipython().system('pip install fuzzywuzzy')
get_ipython().system('pip install python-Levenshtein')


# In[9]:


# Import necessary libraries
import pandas as pd

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Load the two data sets
df1 = pd.read_csv('amz_com-ecommerce_sample.csv',encoding='unicode_escape')
df2 = pd.read_csv('flipkart_com-ecommerce_sample.csv',encoding='unicode_escape')

# Extract the product names, retail prices, and discount prices from the data sets
products1 = df1['product_name'].tolist()
retail_prices1 = df1['retail_price'].tolist()
discount_prices1 = df1['discounted_price'].tolist()
products2 = df2['product_name'].tolist()
retail_prices2 = df2['retail_price'].tolist()
discount_prices2 = df2['discounted_price'].tolist()

# Prompt the user for a product name
product_name = input('Enter a product name: ')

# Find the index of the product in data set 1 with the given name
index1 = products1.index(product_name)

# Use fuzzy string matching to find the most similar product name in data set 2
similar_products = process.extract(product_name, products2, limit=1)
similar_product_name = similar_products[0][0]

# Find the index of the most similar product in data set 2
index2 = products2.index(similar_product_name)

# Print the product details in a tabular column format
print('Product Name\tRetail Price\tDiscount Price')
print(product_name + '\t' + str(retail_prices1[index1]) + '\t' + str(discount_prices1[index1]))
print(similar_product_name + '\t' + str(retail_prices2[index2]) + '\t' + str(discount_prices2[index2]))


# In[18]:


#!pip install bert-tensorflow
#!pip install bert-serving-server
get_ipython().system('pip install bert-serving-client')


# To reduce the time complexity of the  code is to use vectorization to compute the cosine similarities between the product name and all the product names from both datasets at once. This will avoid the need to iterate over the product names and compute the cosine similarity for each one individually.

# In[ ]:


from bert_serving.client import BertClient
from scipy.spatial.distance import cosine
import pandas as pd

# Read in the datasets
dataset1 = pd.read_csv("amz_com-ecommerce_sample.csv", encoding="unicode_escape")
dataset2 = pd.read_csv("flipkart_com-ecommerce_sample.csv", encoding="unicode_escape")

# Get the product names from the datasets
product_names1 = dataset1["product_name"]
product_names2 = dataset2["product_name"]

# Prompt the user for a product name
product_name = input("Enter a product name: ")

# Connect to the BERT server
bc = BertClient()

# Encode the product name and all the product names from the datasets using BERT
product_vec = bc.encode([product_name])
product_vecs1 = bc.encode(product_names1)
product_vecs2 = bc.encode(product_names2)

# Compute the cosine similarity between the encoded product name and all the encoded product names from the first dataset
cosine_similarities1 = 1 - cosine(product_vec, product_vecs1)

# Find the product name from the first dataset with the highest cosine similarity
most_similar_index1 = cosine_similarities1.argmax()
most_similar_product1 = product_names1[most_similar_index1]

# Create a DataFrame to store the results
results = pd.DataFrame(columns=["Product Name", "Retail Price", "Discounted Price"])

# Add the most similar product from the first dataset to the DataFrame
results = results.append({"Product Name": most_similar_product1, 
                         "Retail Price": dataset1["retail_price"][most_similar_index1],
                         "Discounted Price": dataset1["discounted_price"][most_similar_index1]}, 
                        ignore_index=True)

# Compute the cosine similarity between the encoded product name and all the encoded product names from the second dataset
cosine_similarities2 = 1 - cosine(product_vec, product_vecs2)

# Find the product name from the second dataset with the highest cosine similarity
most_similar_index2 = cosine_similarities2.argmax()
most_similar_product2 = product_names2[most_similar_index2]

# Add the most similar product from the second dataset to the DataFrame
results = results.append({"Product Name": most_similar_product2, 
                         "Retail Price": dataset2["retail_price"][most_similar_index2],
                         "Discounted Price": dataset2["discounted_price"][most_similar_index2]}, 
                        ignore_index=True)
results


# In[ ]:


sds


# In[ ]:


from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import pandas as pd

# Read in the datasets
dataset1 = pd.read_csv("amz_com-ecommerce_sample.csv", encoding="unicode_escape")
dataset2 = pd.read_csv("flipkart_com-ecommerce_sample.csv", encoding="unicode_escape")

# Get the product names from the datasets
product_names1 = dataset1["product_name"]
product_names2 = dataset2["product_name"]

# Train a word2vec model on the product names
model = Word2Vec(product_names1, min_count=1)

# Prompt the user for a product name
product_name = input("Enter a product name: ")

# Get the vector for the user-specified product name
product_vec = model.wv[product_name]

# Initialize a list to store the cosine similarities
cosine_similarities = []

# Iterate over the product names from the first dataset
for name in product_names1:
    # Get the vector for the current product name
    name_vec = model.wv[name]

    # Compute the cosine similarity between the vectors
    sim = 1 - cosine(product_vec, name_vec)

    # Store the cosine similarity in the list
    cosine_similarities.append(sim)

# Find the product name from the first dataset with the highest cosine similarity
most_similar_index = cosine_similarities.index(max(cosine_similarities))
most_similar_product = product_names1[most_similar_index]

# Create a DataFrame to store the results
results = pd.DataFrame(columns=["Product name in Flipkart", "Retail Price in Flipkart", "Discounted Price in Flipkart",
                                "Product name in Amazon", "Retail Price in Amazon", "Discounted Price in Amazon"])

# Add the most similar product from the first dataset to the DataFrame
results = results.append({"Product name in Amazon": most_similar_product, 
                         "Retail Price in Amazon": dataset1["retail_price"][most_similar_index],
                         "Discounted Price in Amazon": dataset1["discounted_price"][most_similar_index],
                         "Product name in Flipkart": "N/A",
                         "Retail Price in Flipkart": "N/A",
                         "Discounted Price in Flipkart": "N/A"}, 
                        ignore_index=True)

# Repeat the process for the product names from the second dataset
# Initialize a list to store the cosine similarities
cosine_similarities = []

# Iterate over the product names from the first dataset
for name in product_names2:
    # Get the vector for the current product name
    name_vec = model.wv[name]

    # Compute the cosine similarity between the vectors
    sim = 1 - cosine(product_vec, name_vec)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





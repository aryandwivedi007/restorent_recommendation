#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[120]:


#reading the dataset
zomato_real=pd.read_csv("./zomato.csv")
zomato_real.head() 


# In[121]:


zomato_real.info()


# In[122]:


#Deleting Unnnecessary Columns
zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) 
#Dropping the column "dish_liked", "phone", "url" and saving the new dataset as "zomato"


# In[123]:


print("Duplicate Rows:", zomato.duplicated().sum())  # Count duplicates
zomato.drop_duplicates(inplace=True)  # Remove duplicates
print("Duplicate Rows After Removal:", zomato.duplicated().sum())  # Verify


# In[124]:


# Count and display missing values before removal
print("Missing Values Before Removal:\n", zomato.isnull().sum())

# Remove rows with NaN values
zomato.dropna(how='any', inplace=True)

# Display missing values after removal
print("\nMissing Values After Removal:\n", zomato.isnull().sum())

# Print dataset summary
print("\nDataset Summary After NaN Removal:")
print(zomato.info())


# In[125]:


#Reading Column Names
zomato.columns


# In[126]:


#Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                                  'listed_in(city)':'city'})
zomato.columns


# In[127]:


#Some Transformations
zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost to '.' eg 1,500 to 1.500
zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float now it converted from 1.500->1500.0
zomato.info()


# In[128]:


zomato['rate'].unique()


# In[129]:


# Removing '/5' from Rates

# Removing rows where 'rate' is 'NEW'
zomato = zomato.loc[zomato.rate != 'NEW']

# Removing rows where 'rate' is '-' and resetting index
zomato = zomato.loc[zomato.rate != '-'].reset_index(drop=True)

# Defining a lambda function to remove '/5' from ratings
remove_slash = lambda x: x.replace('/5', '') if isinstance(x, str) else x

# Applying the lambda function to 'rate' column, stripping spaces, and converting to float
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')

# Displaying first few values of 'rate' after transformation
zomato['rate'].head()


# In[130]:


# Adjust the column names

# Capitalizing the first letter of each word in the 'name' column
zomato.name = zomato.name.apply(lambda x: x.title())

# Replacing 'Yes' with True and 'No' with False in the 'online_order' column
zomato.online_order.replace(('Yes', 'No'), (True, False), inplace=True)

# Replacing 'Yes' with True and 'No' with False in the 'book_table' column
zomato.book_table.replace(('Yes', 'No'), (True, False), inplace=True)

# Displaying unique values in the 'cost' column to check the range of different costs
zomato.cost.unique()


# In[131]:


zomato.head()


# In[132]:


zomato['city'].unique() #retrieves and displays all unique values in the 'city' column of the zomato DataFrame.


# In[133]:


## Checking Null values
zomato.isnull().sum()


# In[134]:


zomato.head()


# In[135]:


# Count the number of duplicate restaurant names
duplicate_counts = zomato.duplicated(subset=['name'], keep=False).sum()
print(f"Number of duplicate restaurant names: {duplicate_counts}")

# Get a list of unique restaurant names
restaurants = list(zomato['name'].unique())

# Create a new column 'Mean Rating' and initialize it with 0
zomato['Mean Rating'] = 0

# Iterate over each unique restaurant name
for i in range(len(restaurants)):
    # Compute the mean rating for each restaurant and assign it
    zomato.loc[zomato['name'] == restaurants[i], 'Mean Rating'] = zomato.loc[zomato['name'] == restaurants[i], 'rate'].mean()

# Log the completion message
print("Mean rating calculation completed!")


# In[136]:


zomato.head()


# In[137]:


# Importing MinMaxScaler from scikit-learn
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler is used for feature scaling (normalization).
# It transforms values into a specified range (default: 0 to 1).
# Here, we set the range between 1 and 5.
scaler = MinMaxScaler(feature_range=(1, 5))

# Applying MinMaxScaler on the 'Mean Rating' column to scale its values between 1 and 5.
# .fit_transform() fits the data and scales it in one step.
# .round(2) ensures values are rounded to 2 decimal places.
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

# Displaying a random sample of 3 rows from the DataFrame to check the transformed values.
zomato.sample(3)


# In[138]:


zomato.head()


# In[139]:


# 5 examples of these columns before text processing:
zomato[['reviews_list', 'cuisines']].sample(5)


# In[140]:


## Lower Casing
zomato["reviews_list"] = zomato["reviews_list"].str.lower()
zomato[['reviews_list', 'cuisines']].sample(5)


# In[141]:


## Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation #!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ provided letters 
# str.translate(x,y,z) 
# x: Characters to replace (not used in our case).
# y: Characters to replace them with (not used in our case).
# z: Characters to remove (we use this to remove punctuation).
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))
zomato[['reviews_list', 'cuisines']].sample(5)


# In[40]:


# Stopwords are commonly used words (like "is," "the," "in," "and") that do not add much meaning to the text.
# Removing stopwords helps in improving text analysis and machine learning models (e.g., sentiment analysis, search optimization).
import nltk
# nltk.download('stopwords')


# In[142]:


## Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))


# In[143]:


## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))


# In[144]:


zomato[['reviews_list', 'cuisines']].sample(5)


# In[145]:


# RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())
restaurant_names


# In[148]:


from sklearn.feature_extraction.text import CountVectorizer

def get_top_words(column, top_nu_of_words, nu_of_word):
    """
    Extracts the top N most frequent words or n-grams from the given text column.
    
    Parameters:
    - column: list or pandas Series containing text data
    - top_nu_of_words: Number of top words to return
    - nu_of_word: Tuple specifying the n-gram range (e.g., (1,1) for unigrams, (2,2) for bigrams)
    
    Returns:
    - List of tuples containing the top N words/n-grams and their frequencies
    """

    vec = CountVectorizer(ngram_range=nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:top_nu_of_words]


# In[149]:


zomato.head()


# In[150]:


zomato.sample(5)


# In[151]:


zomato=zomato.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)


# In[152]:


zomato.columns


# In[ ]:





# In[153]:


import pandas

# Randomly sample 50% of your dataframe
df_percent = zomato.sample(frac=0.5)


# In[154]:


df_percent.shape


# In[155]:


df_percent.set_index('name', inplace=True)


# In[156]:


indices = pd.Series(df_percent.index)


# In[157]:


# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.000001, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])


# In[158]:


cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[104]:


print(cosine_similarities)


# In[159]:

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from difflib import get_close_matches

valid_restaurant_names = df_percent.index.unique().tolist()



def get_closest_matches(query, valid_names, limit=5):
    """Returns closest matches for a given query from valid restaurant names."""
    return get_close_matches(query, valid_names, n=limit, cutoff=0.4)  # Adjust cutoff if needed


def recommend(name, cosine_similarities=cosine_similarities):
    # Create a list to store top recommended restaurants
    recommend_restaurant = []
    
    # Find the index of the restaurant entered
    idx = indices[indices == name].index[0]
    
    # Find restaurants with similar cosine similarity values and sort them in descending order
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract the top 30 restaurant indexes with similar cosine similarity values
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Get the names of these top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Create a new DataFrame to store similar restaurant details
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Fetch details of the top 30 similar restaurants
    for each in recommend_restaurant:
        df_new = pd.concat([df_new, df_percent[['cuisines', 'Mean Rating', 'cost']][df_percent.index == each].sample()])
    
    # Drop duplicate restaurants and select the top 10 based on rating
    df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print(f'TOP {len(df_new)} RESTAURANTS LIKE {name} WITH SIMILAR REVIEWS:')
    
    return df_new


# In[160]:


# HERE IS A RANDOM RESTAURANT. LET'S SEE THE DETAILS ABOUT THIS RESTAURANT:
df_percent[df_percent.index == 'Pai Vihar'].head()


# In[161]:


recommend('The Marash')


# In[105]:




@app.route('/api/suggest', methods=['GET'])
def get_suggestions():
    query = request.args.get('name', '').strip().lower()
    
    if not query:
        return jsonify([])  # Return empty list if no query

    # Find matching restaurant names
    matches = get_closest_matches(query, valid_restaurant_names, limit=7)
    
    return jsonify(matches)



@app.route('/api/recommend', methods=['GET'])
def get_recommendation():
    restaurant_name = request.args.get('name')

    if not restaurant_name:
        return jsonify({"error": "Please provide a restaurant name"}), 400

    recommendations = recommend(restaurant_name)  # Get recommendations

    if recommendations is None or recommendations.empty:  # Check if DataFrame is empty
        return jsonify({"error": f"No data found for restaurant name: '{restaurant_name}'"}), 404

    try:
        recommendations_json = recommendations.to_dict(orient="records")  # Convert to JSON
        return jsonify(recommendations_json)
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500  # Catch any unexpected errors


if __name__ == '__main__':
    app.run(debug=True)





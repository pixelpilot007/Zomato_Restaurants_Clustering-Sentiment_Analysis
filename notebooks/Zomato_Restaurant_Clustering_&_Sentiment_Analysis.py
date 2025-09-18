#!/usr/bin/env python
# coding: utf-8

# # ***Zomato Restaurant Clustering and Sentiment Analysis***

# ### **Project Overview**
# 
# This project leverages data from Zomato, India's leading food technology platform established by Deepinder Goyal and Pankaj Chaddah in 2008. Zomato operates as a comprehensive restaurant discovery service that aggregates dining establishments, provides menu information, customer ratings, and facilitates food delivery partnerships across major Indian cities.
# 
# India's rich culinary heritage is reflected in its vast array of dining establishments and food service venues, showcasing the country's cultural diversity through cuisine. The restaurant industry in India is experiencing continuous growth as consumers increasingly embrace dining out and food delivery services. The expanding network of restaurants across Indian states offers valuable opportunities to analyze market data and derive insights about the food service landscape in different cities. This study focuses on examining Zomato's restaurant database to understand clustering patterns and customer sentiment across Indian markets.
# 
# **Dataset Overview:**
# 
# The analysis employs two complementary datasets with descriptive column structures:
# 
# 1. **Restaurant Profiles Metadata** - Contains establishment metadata ideal for segmentation and clustering analysis. This dataset provides detailed information about cuisine types, pricing tiers, and operational attributes, enabling market categorization and competitive positioning studies.
# 
# 2. **Restaurants Review Data** - Features user-generated content perfect for sentiment analysis and opinion mining. The dataset includes reviewer profile information and engagement metrics, allowing for the identification of influential food reviewers and industry thought leaders.
# 
# **Project Scope:**
# This initiative aims to discover restaurant clusters based on operational characteristics while simultaneously analyzing customer sentiment patterns to provide comprehensive insights into India's restaurant ecosystem through advanced data mining and natural language processing techniques.

# In[415]:


from IPython.display import Image, display

display(Image(filename='zomato-poster.png',width=1000))


# ### Major steps
# - Know your data
# - Data Wrangling
# - Understanding dataset variable
# - Data Vizualization & Experimenting with charts
# - Understand the relationships between variables
# - Text preprocessing on both datasets
# - LDA (Latent Dirichlet Allocation)
# - Sentiment Analysis
# - Clustering
# - K-means Clustering
# - Conclusion
# 
# 
# 

# # 1. Know your data

# ### Importing libraries

# In[416]:


# Import Libraries and modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from textblob import TextBlob
from IPython.display import Image
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import gensim

import warnings
warnings.filterwarnings('ignore')


# In[417]:


# Loading Zomato Restaurant names and Metadata Dataset 
restaurant_metadata_main=pd.read_csv('Restaurant names and Metadata.csv')


# In[418]:


restaurant_metadata=restaurant_metadata_main.copy()

#Loading Zomato Restaurant reviews Dataset
review=pd.read_csv('Zomato Restaurant reviews.csv')


# ### Dataset first view
# 

# In[419]:


# Restuarant Metadata Data
restaurant_metadata.head()


# In[420]:


# Review of Restuarant Metadata
review.head()


# ### Dataset Rows & Columns count

# In[421]:


# Size of Restuarant Metadata Dataset
restaurant_metadata.shape


# In[422]:


# Size of Review Dataset
review.shape


# ### Datasets Information
# 

# Ensures data is clean before processing.

# In[423]:


# Restuarant Metadata Info
restaurant_metadata.info()


# In[424]:


# Review Info
review.info()


# ### Describing Datasets

# In[425]:


restaurant_metadata.describe(include='all')


# In[426]:


review.describe(include='all')


# # 2. Data Wrangling

# Data wrangling (also called data munging) is the process of cleaning, transforming, and preparing raw data into a usable format for analysis, machine learning, or visualization.

# ### Duplicate Values Count

# In[427]:


# Restuarant Metadata Duplicate Value Count
print(len(restaurant_metadata[restaurant_metadata.duplicated()]))


# In[428]:


# Review Datasaet Duplicate Value Count
print(len(review[review.duplicated()]))


# In[429]:


# List the duplicate values present in Review dataset
review[review.duplicated()]


# ### Missing/Null Values Check

# Preparing to calculate nulls.

# **pandas:** .isnull() → marks NaN as True.
# 
# **Formula:**
# 
# $
# Missing\ Count(col) = \sum_{i=1}^n 1[value_i = NaN]
# $
# 
# Counts missing values per column in restaurant dataset.

# In[430]:


# Missing Values/Null Values Count in restaurant_metadata
restaurant_metadata.isnull().sum()


# In[431]:


# Visualizing the missing values
plt.figure(figsize=(10,5))
sns.heatmap(restaurant_metadata.isnull(),cmap='cividis',annot=False,yticklabels=False)
plt.title(" Visualising Missing Values");


# In[432]:


# Missing Values/Null Values Count in review
review.isnull().sum()


# In[433]:


# Visualizing the missing values
plt.figure(figsize=(10,5))
sns.heatmap(review.isnull(),cmap='YlGnBu',annot=False,yticklabels=False)
plt.title(" Visualising Missing Values");


# #### What all we have concluded from our data till now

# # 3. Understanding dataset variable

# ### Datasets Columns

# In[434]:


restaurant_metadata.columns


# In[435]:


review.columns


# ### Dataset Variables Description

# #### Zomato Metadata Dataset 
# - Name : Name of Restaurants
# 
# - Links : URL Links of Restaurants
# 
# - Cost : Per person estimated Cost of dining
# 
# - Collection : Tagging of Restaurants w.r.t. Zomato categories
# 
# - Cuisines : Cuisines served by Restaurants
# 
# - Timings : Restaurant Timings

# #### Zoamto Review Dataset
# 
# - Restaurant : Name of the Restaurant
# 
# - Reviewer : Name of the Reviewer
# 
# - Review : Review Text
# 
# - Rating : Rating Provided by Reviewer
# 
# - MetaData : Reviewer Metadata - No. of Reviews and followers
# 
# - Time: Date and Time of Review
# 
# - Pictures : No. of pictures posted with review

# ### Data Wrangling on "restaurant_metadata"

# In[436]:


# Convert the 'Cost' column, deleting the comma and changing the data type into 'int64'.
restaurant_metadata["Cost"] = restaurant_metadata["Cost"].str.replace(",","").astype('int64')
restaurant_metadata.info()


# # 4. Data Vizualization & Experimenting with charts : Understand the relationships between variables

# ##### Moving into EDA (Exploratory Data Analysis)
# Exploratory Data Analysis (EDA) in Python is the process of analyzing datasets to summarize their main characteristics, discover patterns, identify outliers, and understand relationships between variables

# ### Chart 1: Word Cloud for Expensive Restaurants

# **Library:** WordCloud
# 
# **Formula:** Word frequency → bigger word = more frequent.
# 
# Helps visualize common keywords in reviews.

# In[517]:


# Chart - 1 visualization code.
top10_res_by_cost = restaurant_metadata[['Name','Cost']].groupby('Name',as_index=False).sum().sort_values(by='Cost',ascending=False).head(10)
     


# In[518]:


# Creating word cloud for expensive restaurants
plt.figure(figsize=(10,6))
text = " ".join(name for name in restaurant_metadata.sort_values('Cost',ascending=False).Name[:30])

# Creating word_cloud with text as argument in .generate() method.
word_cloud = WordCloud(width = 1400, height = 1400,collocations = False, background_color = 'pink').generate(text)

# Display the generated Word Cloud.
plt.imshow(word_cloud, interpolation='bilinear')

plt.axis("off");


# ### Chart 2: Word Cloud for Cheapest Restaurants

# In[519]:


# Creating word cloud for cheapest restaurants
plt.figure(figsize=(8,6))
text = " ".join(name for name in restaurant_metadata.sort_values('Cost',ascending=True).Name[:30])

# Creating word_cloud with text as argument in .generate() method
wordcloud = WordCloud(background_color="lemonchiffon").generate(text)

# Display the generated Word Cloud
plt.imshow(wordcloud, interpolation='bilinear');

plt.axis("off")


# ### Chart 3

# In[440]:


# Affordable price restaurants.

plt.figure(figsize=(6,5))

# Performing groupby To get values accourding to Names and sort it for visualisation.
top_10_affor_rest=restaurant_metadata[['Name','Cost']].groupby('Name',as_index=False).sum().sort_values(by='Cost',ascending=False).tail(10)

# Lables for X and Y axis
x = top_10_affor_rest['Cost']
y = top_10_affor_rest['Name']

# Assigning the arguments for chart
plt.title("Top 10 Affordable Restaurant",fontsize=15, weight='bold',color=sns.cubehelix_palette(8, start=.5, rot=-.75)[-3])
plt.ylabel("Name",weight='bold',fontsize=10)
plt.xlabel("Cost",weight='bold',fontsize=10)
plt.xticks(rotation=90)
sns.barplot(x=x, y=y,palette='viridis')
plt.show()


# ### Chart 4

# In[441]:


# Visualisation the value counts of collection. 
restaurant_metadata['Collections'].value_counts()[0:10].sort_values().plot(figsize=(6,5),kind='barh',color=sns.color_palette("mako_r",10))


# # 5. Text Preprocessing  
# We prepare the reviews for NLP by cleaning text:  
# - Lowercasing  
# - Removing punctuation/numbers  
# - Removing unwanted characters  
# - Removing stopwords  
# - Tokenization  
# - Stemming/Lemmatization

# ### 5.1 Text preprocessing on "restaurant_metadata" Dataset

# In Order to plot the cuisines from the data we have to count the frequency of the words from the document.(Frequency of cuisine). For that We have to perform the opration like removing stop words, Convert all the text into lower case, removing punctuations, removing repeated charactors, removing Numbers and emojies and finally count vectorizer.

# **Library used:** `nltk`  
# 
# - `word_tokenize()` → splits review text into words.  
# - `stopwords.words('english')` → removes common filler words ("the", "is").  
# 
# **Formula:**  
# $
# Tokens = [w \in Review \ | \ w \notin Stopwords]
# $

# **Stemming:** Reduces a word to its base/root form.  
# 
# Example:  
# - "cooking", "cooked", "cooks" → "cook"  
# 
# **Formula:**  
# $
# Stem(w) = Root(w)
# $

# In[520]:


# Downloading and importing the dependancies for text cleaning.
nltk.download('stopwords')
from nltk.corpus import stopwords


# Stopwords are common, high-frequency words like "the," "and," and "is" that are filtered out before text analysis because they typically carry little meaning on their own.

# In[443]:


# Extracting the stopwords from nltk library for English corpus.
sw = stopwords.words('english')


# ### Removing Stopwords

# In[444]:


# Creating a function for removing stopwords.
def stopwords(text):
    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in str(text).split() if word.lower() not in sw]

    # joining the list of words with space separator
    return " ".join(text)
     


# In[445]:


# Removing stopwords from Cuisines.
restaurant_metadata['Cuisines'] = restaurant_metadata['Cuisines'].apply(lambda text: stopwords(text))
restaurant_metadata['Cuisines'].head()


# Stop words are removed successfully

# ### Removing Punctuation

# In[446]:


# Defining the function for removing punctuation.
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string

    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    
    # return the text stripped of punctuation marks
    return text.translate(translator)


# In[447]:


# Removing punctuation from Cuisines.
restaurant_metadata['Cuisines'] = restaurant_metadata['Cuisines'].apply(lambda x: remove_punctuation(x))
restaurant_metadata['Cuisines'].head()


# Punctuations present in the text are removed successfully

# ### Removing Repeated Characters

# In[448]:


# Cleaning and removing repeated characters.
import re

# Writing a function to remove repeating characters.
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
     


# In[449]:


# Removing repeating characters from Cuisines.
restaurant_metadata['Cuisines'] = restaurant_metadata['Cuisines'].apply(lambda x: cleaning_repeating_char(x))
restaurant_metadata['Cuisines'].head()


# Removed repeated characters successfully

# ### Removing Numbers

# In[450]:


# Defining a function to remove numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
     


# In[451]:


# Removing numbers
restaurant_metadata['Cuisines'] = restaurant_metadata['Cuisines'].apply(lambda x: cleaning_numbers(x))
restaurant_metadata['Cuisines'].head()


# We don't want numbers in the text, hence removed number successfully

# In[452]:


# Top 20 Two word Frequencies of Cuisines.
from collections import Counter 
text = ' '.join(restaurant_metadata['Cuisines'])

# separating each word from the sentences
words = text.split()

# Extracting the first word from the number for cuisines in the sentence.
two_words = {' '.join(words):n for words,n in Counter(zip(words, words[1:])).items() if not  words[0][-1]==(',')}


# In[453]:


# Extracting the most frequent cuisine present in the collection.
# Counting a frequency for cuisines.
two_words_dfc = pd.DataFrame(two_words.items(), columns=['Cuisine Words', 'Frequency'])

# Sorting the most frequent cuisine at the top and order by descending
two_words_dfc = two_words_dfc.sort_values(by = "Frequency", ascending = False)

# selecting first top 20 frequent cuisine.
two_words_20c = two_words_dfc[:20]
two_words_20c


# ### Chart 5

# In[454]:


# Visualizing the frequency of the Cuisines.

sns.set_style("whitegrid")
plt.figure(figsize = (11, 6))
sns.barplot(y = "Cuisine Words", x = "Frequency", data = two_words_20c, palette = "magma")
plt.title("Top 20 Two-word Frequency of Cuisines", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel("Cuisine Words", size = 20)
plt.ylabel(None)
plt.savefig("Top_20_Two-word_Frequencies_of_Cuisines.png")
plt.show()


# The DataFrame contains two columns: "Cuisine Words" and "Frequency." The "Cuisine Words" column lists the most frequent two-word cuisine terms, while the "Frequency" column shows the number of times each two-word cuisine term appears in the dataset.This information can be helpful in understanding the most common cuisine types in the dataset. It can also be used to identify trends and patterns in the types of cuisines that are popular or in demand among the customers.

# ### 5.2 Text preprocessing on "review" Dataset

# ### Data Wrangling on "review"

# In[455]:


# proportion or percentage of occurrences for each unique value in the Rating column.
review['Rating'].value_counts(normalize=True)


# In[456]:


# Removing like value and taking the mean in the rating column.
review.loc[review['Rating'] == 'Like'] = np.nan

 # Chenging the data type of rating column 
review['Rating']= review['Rating'].astype('float64')

print(review['Rating'].mean())


# In[457]:


# Filling mean in place of null value
review['Rating'].fillna(3.6, inplace=True)


# In[458]:


# Changing the data type of review column.
review['Review'] = review['Review'].astype(str)

# Creating a review_length column to check the frequency of each rating.
review['Review_length'] = review['Review'].apply(len)


# In[459]:


review['Rating'].value_counts(normalize=True)


# The Ratings distribution 38% reviews are 5 rated,23% are 4 rated stating that people do rate good food high.

# ### Chart 6

# In[460]:


# Visualizing the rating column against the review length.
# Polting the frequency of the rating on scatter bar plot

import plotly.express as px
fig = px.scatter(review, x=review['Rating'], y=review['Review_length'])
fig.update_layout(title_text="Rating vs Review_Length", width=800, height=500)
fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=0, ticklen=10)
fig.show()


# The scatter plot confirms that length of review doesn't impact ratings.

# ### Chart 7

# In[461]:


# Creating polarity variable to see sentiments in reviews.(using textblob) 
from textblob import TextBlob
review['Polarity'] = review['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[462]:


# Visualizing the polarity using histogram.
review['Polarity'].plot(kind='hist', bins=100,color="violet")


# Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement. Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. Subjectivity is also a float which lies in the range of [0,1].

# ### Removing Stop Words

# Stop words are used in a language to removed from text data during natural language processing. This helps to reduce the dimensionality of the feature space and focus on the more important words in the text.

# In[463]:


# Importing dependancies and removing stopwords.

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Creating argument for stop words.
stop_words = stopwords.words('english')

print(stop_words)
rest_word=['order','restaurant','taste','ordered','good','food','table','place','one','also']
rest_word


# ### Removing Special Characters

# In[464]:


# Removing Special characters and punctuation from review columns.

import re
review['Review']=review['Review'].map(lambda x: re.sub('[,\.!?]','', x))
review['Review']=review['Review'].map(lambda x: x.lower())
review['Review']=review['Review'].map(lambda x: x.split())
review['Review']=review['Review'].apply(lambda x: [item for item in x if item not in stop_words])
review['Review']=review['Review'].apply(lambda x: [item for item in x if item not in rest_word])


# ### Chart 8

# In[465]:


# We will extrapolate the 15 profiles that have made more reviews.

# Groupby on the basis of rivewer gives the fequency of the reviews
reviewer_list = review.groupby('Reviewer').apply(lambda x: x['Reviewer'].count()).reset_index(name='Review_Count')

 # Sorting the frequency of reviews decending
reviewer_list = reviewer_list.sort_values(by = 'Review_Count',ascending=False)

# Selecting the top 15 reviewrs
top_reviewers = reviewer_list[:15]


# In[466]:


# Visualizing the top 15 reviewers.
plt.figure(figsize=(10,5))
plt.bar(top_reviewers['Reviewer'], top_reviewers['Review_Count'], color = sns.color_palette("pastel", 8))
plt.xticks(rotation=75)
plt.title('Top 15 reviews',size=20)
plt.xlabel("Reviewer's Name",size=15)
plt.ylabel('N of reviews',size=15)


# ### Chart 9

# In[467]:


# Calculate the average of their ratings review.
review_ratings=review.groupby('Reviewer').apply(lambda x:np.average(x['Rating'])).reset_index(name='Average_Ratings')
review_ratings=pd.merge(top_reviewers,review_ratings,how='inner',left_on='Reviewer',right_on='Reviewer')
top_reviewers_ratings=review_ratings[:15]


# In[468]:


# Average rating of top reviewers.
plt.figure(figsize=(11,6))
x = top_reviewers_ratings['Average_Ratings']
y = top_reviewers_ratings['Reviewer']
plt.title("Top 15 reviewers with average rating of review",fontsize=20, weight='bold',color=sns.cubehelix_palette(8, start=.5, rot=90)[-5])
plt.ylabel("Name",weight='bold',fontsize=15)
plt.xlabel("Average Ratings",weight='bold',fontsize=15)
plt.xticks(rotation=90)
sns.barplot(x=x, y=y,palette='plasma')
plt.show()


# The output of top 15 reviewers based on the number of reviews they have made in a given dataset. Analyzing the reviews made by these top reviewers can help in improving customer satisfaction and loyalty, ultimately leading to increased revenue and growth.

# ### Chart 10: Word cloud for Positive Reviews

# **Summary:** Creates a word cloud for positive reviews
# 
# **Explanation:**
# - WordCloud (from wordcloud library) generates a cloud where word size = frequency of appearance.
# - Used here to visualize most common words in positive reviews.
# 
# **Formula/Logic:** WordCloud(text).generate(text_data) → creates frequency-based visualization.
# 

# In[469]:


# Word cloud for positive reviews.

from wordcloud import WordCloud
review['Review']=review['Review'].astype(str)

ps = PorterStemmer() 
review['Review']=review['Review'].map(lambda x: ps.stem(x))
long_string = ','.join(list(review['Review'].values))
long_string
wordcloud = WordCloud(background_color="lightyellow", max_words=100, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# Service, taste, food, place are the key to good review.

# ### Chart 11: Word cloud for Negative Reviews

# **Summary:** Creates a word cloud for negative reviews
# 
# **Explanation:**
# - Similar to previous cell but applied to negative reviews.
# 
# **Formula/Logic:** WordCloud(text).generate(text_data) → word cloud for negative sentiment.
# 

# In[470]:


# Creating two datasets for positive and negative reviews.

review['Rating']= pd.to_numeric(review['Rating'],errors='coerce')   # The to_numeric() function in pandas is used to convert a pandas object to a numeric type.
pos_rev = review[review.Rating>= 3]
neg_rev = review[review.Rating< 3]
     


# In[471]:


# Negative reviews wordcloud.

long_string = ','.join(list(neg_rev['Review'].values))
long_string
wordcloud = WordCloud(background_color="wheat", max_words=100, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# Service, bad chicken, staff behavior, quality are the key reasons for neagtive reviews

# ### Text Cleaning

# In[472]:


# Creating word embeddings (for positive and negative reviews).

from gensim.models import word2vec
pos_rev = review[review.Rating>= 3]
neg_rev = review[review.Rating< 3]


# Dataframe where the Rating column is greater than or equal to 3. This selects all the positive reviews where as the Rating column is less than 3. This selects all the negative reviews, assuming that the Rating column is a scale from 1 to 5 with 5 being the highest rating.

# ### Create a corpus of words from the positive reviews in the neg_rev DataFrame.

# In[473]:


# Plot for postive reviews
def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['Review']:
        for sentence in data[col].items():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus
    
# Display the first two elements of the corpus list
corpus = build_corpus(pos_rev)        
corpus[0:2]
     


# ### Create a corpus of words from the negative reviews in the neg_rev DataFrame.

# In[474]:


# Plot for negative reviews.
def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['Review']:
        for sentence in data[col].items():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus
    
# Display the first two elements of the corpus list
corpus = build_corpus(neg_rev)        
corpus[0:2]


# In[475]:


# Checking for the implimented code
review['Review']


# # 7. LDA (Latent Dirichlet Allocation)

# ##### What is LDA?
# - **LDA (Latent Dirichlet Allocation)** is a **topic modeling algorithm**.
# - It automatically finds **hidden topics** in a collection of documents (reviews).
# - Each review is assumed to be a mixture of topics, and each topic is a collection of words.
# - Example:
#   - Review: "The delivery was late but food was tasty"
#   - LDA might assign:
#     - 60% to **Delivery Topic** (words: late, delivery, time)
#     - 40% to **Food Topic** (words: tasty, food, delicious)
# 
# ---
# 
# ##### Why LDA in Zomato Review Analysis?
# - Helps discover **main themes** people talk about:
#   - "Food Quality"
#   - "Delivery Speed"
#   - "Ambience"
#   - "Price"
# - When combined with **sentiment analysis + clustering**:
#   - Cluster reviews into **positive/negative/neutral**.
#   - Apply **LDA** on each cluster.
#   - Identify **which topics drive positivity or negativity**.
# 
# **Example:**
# - Positive cluster → Topics: *"delicious food"*, *"great ambience"*
# - Negative cluster → Topics: *"late delivery"*, *"poor service"*
# 
# 
# 

# **Summary:** Imports libraries for topic modeling (LDA)
# 
# **Explanation:**
# - gensim.corpora.Dictionary → converts text into word IDs.
# - gensim.models.LdaModel → Latent Dirichlet Allocation (LDA) for topic modeling.
# **Formula/Logic:**
# 
# **LDA:** Each document = mixture of topics, each topic = mixture of words.

# In[476]:


from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess


# Listing the top 10 most occuring words. Topic modeling is a process to automatically identify topics present in a text object and to assign text corpus to one category of topic.
# 
# 

# In[477]:


# Assume that documents is a list of strings representing text documents

# Tokenize the documents
tokenized_docs = [simple_preprocess(doc) for doc in review['Review']]

# Create a dictionary from the tokenized documents
dictionary = corpora.Dictionary(tokenized_docs)

# Convert the tokenized documents to a bag-of-words corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Train an LDA model on the bag-of-words corpus
num_topics = 10  # The number of topics to extract
lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print the topics and their top 10 terms
for topic in lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False):
    print('Topic {}: {}'.format(topic[0], ', '.join([term[0] for term in topic[1]])))


# In[478]:


import gensim 
import pyLDAvis.gensim 
import pyLDAvis.lda_model 
pyLDAvis.enable_notebook()
     


# In[479]:


lda_visualization = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, mds='tsne') 
pyLDAvis.display(lda_visualization)


# The topics and topic terms can be visualised to help assess how interpretable the topic model is.

# # 8. Sentiment Analysis

# ### TF-IDF Vectorization
# 
# We convert cleaned reviews into numerical form using TF-IDF.
# 
# **Library used:** `sklearn.feature_extraction.text.TfidfTransformer`  
# 
# **Formula:**  
# $
# TF\text{-}IDF(t,d) = TF(t,d) \times \log \frac{N}{DF(t)}
# $ 
# 
# Where:  
# - \(TF(t,d)\) = frequency of term *t* in document *d*  
# - \(DF(t)\) = number of documents containing *t*  
# - \(N\) = total number of documents

# ### Bag of Words (CountVectorizer)
# 
# We also try Bag of Words, which counts the occurrences of each word.
# 
# **Library used:** `sklearn.feature_extraction.text.CountVectorizer`  
# 
# **Formula:**  
# $
# Count(t,d) = \sum 1[word_i = t]
# $

# In[480]:


from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import plotly.express as px


# In[481]:


# Create a function to get the subjectivity
def subjectivity(text): 
    return TextBlob(text).sentiment.subjectivity


# In[482]:


# Create a function to get the polarity
def polarity(text): 
    return TextBlob(text).sentiment.polarity


# In[483]:


# Applying subjectivity and the polarity function to the respective columns
review['Subjectivity'] = review['Review'].apply(subjectivity)
review['Polarity'] = review['Review'].apply(polarity)


# In[484]:


# Checking for created columns
review['Polarity']


# In[485]:


# Checking for created columns
review['Subjectivity']


# ### Chart 12

# In[486]:


# Create a function to compute the negative, neutral and positive analysis
def getAnalysis(score):
    if score <0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


# If the score is less than 0, the function returns the string 'Negative'. If the score is equal to 0, the function returns the string 'Neutral'. If the score is greater than 0, the function returns the string 'Positive'.

# In[487]:


# Apply get analysis function to separate the sentiments from the column
review['Analysis'] = review['Polarity'].apply(getAnalysis)


# **Summary:** Calculates sentiment distribution (count of each sentiment type)
# 
# **Explanation:**
# - value_counts() → counts how many reviews fall under Positive, Negative, Neutral.
# - normalizes=True → gives proportion instead of count.
# 
# **Formula/Logic:**
# -Sentiment Distribution = count(Sentiment) / total_reviews
# 

# In[488]:


# plot the polarity and subjectivity
plt.figure(figsize=(5,8))
fig = px.scatter(review, 
                 x='Polarity', 
                 y='Subjectivity', 
                 color = 'Analysis',
                 size='Subjectivity',
                 width=900,       
                 height=600,
                color_discrete_map={  
                     "Positive": "seagreen",  
                     "Negative": "crimson",  
                     "Neutral":  "royalblue"   
                 })

# Add a vertical line at x=0 for Netural Reviews
fig.update_layout(title='Sentiment Analysis',
                  shapes=[dict(type= 'line',
                               yref= 'paper', y0= 0, y1= 1, 
                               xref= 'x', x0= 0, x1= 0)])
fig.show()


# The resulting plot can provide several insights into the sentiment analysis results. Firstly, the histogram bars on the left side of the plot (negative polarity) indicate that a significant number of reviews expressed negative sentiments. Similarly, the histogram bars on the right side of the plot (positive polarity) indicate that a significant number of reviews expressed positive sentiments.
# 
# Overall, this plot can provide a quick and easy way to visualize the sentiment polarity distribution of the reviews, which can help in understanding the overall sentiment of the customers towards the restaurants.

# # 9. Clustering

# In[489]:


warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning);


# In[490]:


# converting the cuisines to lower case

restaurant_metadata_main['Cuisines'] = restaurant_metadata_main['Cuisines'].apply(lambda x : x.lower());


# In[491]:


# Separating the Name, cost and cuisines column.
cuisine_df = restaurant_metadata_main.loc[:,['Name','Cost','Cuisines']]


# In[492]:


# Overview of separated variables.
cuisine_df.head()


# In[493]:


# Removing spces from cuisine column.
cuisine_df['Cuisines'] = cuisine_df['Cuisines'].str.replace(' ','')

# Spliting the Words in cuisine.
cuisine_df['Cuisines'] = cuisine_df['Cuisines'].str.split(',')


# In[494]:


# Overview on text cleaning.
cuisine_df.head()


# In[495]:


from sklearn.preprocessing import MultiLabelBinarizer

# converting a list of labels for each sample into a binary indicator matrix
mlb = MultiLabelBinarizer(sparse_output=True)

# converting the Cuisines column in the cuisine_df DataFrame into a binary indicator matrix.
cuisine_df = cuisine_df.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(cuisine_df.pop('Cuisines')),
                                                               index=cuisine_df.index, columns=mlb.classes_)) 


# In[496]:


# Overview
cuisine_df.head()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # adjust as needed


# In[497]:


# Checking the unique for rating.
review['Rating'].unique()
  


# In[498]:


# Remove nan rating in Rating column.
review.dropna(subset=['Rating'],inplace=True)


# In[499]:


# Change data type of rating column to float.
review['Rating']= review['Rating'].astype('float')


# In[500]:


# Dropping the null Values from review column.
review.dropna(subset =['Review'], inplace=True)


# In[501]:


# Grouping the restaurant on the basis of average rating.
ratings_df = review.groupby('Restaurant')['Rating'].mean().reset_index()


# In[502]:


# Top highly rated 15 restaurants.
ratings_df .sort_values(by='Rating',ascending = False).head(15)


# In[503]:


#  Combining the information on restaurant cuisine and ratings into a single DataFrame.
cluster_df = cuisine_df.merge(ratings_df, left_on='Name',right_on='Restaurant')


# In[504]:


# Overview
cluster_df.head()


# ###  Changing name and order of columns

# In[505]:


# List of desired columns in order
cols = ['Name', 'Cost','Rating', 'american', 'andhra', 'arabian', 'asian', 'bbq',
       'bakery', 'beverages', 'biryani', 'burger', 'cafe', 'chinese',
       'continental', 'desserts', 'european', 'fastfood', 'fingerfood', 'goan',
       'healthyfood', 'hyderabadi', 'icecream', 'indonesian', 'italian',
       'japanese', 'juices', 'kebab', 'lebanese', 'malaysian', 'mediterranean',
       'mexican', 'mithai', 'modernindian', 'momos', 'mughlai', 'northeastern',
       'northindian', 'pizza', 'salad', 'seafood', 'southindian', 'spanish',
       'streetfood', 'sushi', 'thai', 'wraps']

# Keep only columns that exist in cluster_df
cols_present = [c for c in cols if c in cluster_df.columns]

# Reorder safely
cluster_df = cluster_df[cols_present]


# In[506]:


# Checking the data type and null counts for newly created variables.
cluster_df.info()


# ### Chart 13

# In[507]:


# Removing commas from the cost variables.
cluster_df['Cost']= cluster_df['Cost'].replace({',': ''}, regex=True)

# Changing the data type of the cost column.
cluster_df['Cost']= cluster_df['Cost'].astype('float')


# In[508]:


# Visualising relationship between the cost of a meal and the rating of a restaurant
sns.lmplot(y='Rating',x='Cost',data=cluster_df,line_kws={'color' :'rebeccapurple'},scatter_kws={'color': 'olive'},height=5, aspect=11.7/10.27)


# The resulting plot shows the relationship between the cost of a meal and the rating of a restaurant, with the regression line indicating the general trend in the data. This can help identify any patterns or correlations between cost and rating.

# # 10. K-means Clustering

# In[509]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer


# ### Chart 14

# In[510]:


# Create a list of inertia scores for different numbers of clusters
scores = [KMeans(n_clusters=i+2, random_state=11).fit(cluster_df.drop('Name',axis=1)).inertia_ 
          for i in range(8)]

# Create a line plot of inertia scores versus number of clusters
plt.figure(figsize=(6,5))
sns.lineplot(x=np.arange(2, 10), y=scores,color="mediumseagreen")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia of k-Means versus number of clusters')
plt.show()
     


# The plot can help to identify the optimal number of clusters based on the elbow point of the curve, where the rate of decrease in inertia score slows down significantly.

# In[511]:


# Initializing a K-Means clustering model with number of clusters and random state.
model = KMeans(random_state=11, n_clusters=5)
model.fit(cluster_df.drop('Name',axis=1))


# In[512]:


# predict the cluster label of a new data point based on a trained clustering model.
cluster_lbl = model.predict(cluster_df.drop('Name',axis=1))
cluster_df['labels'] = cluster_lbl


# In[513]:


# Creating the data frame for each cluster.
cluster_0 = cluster_df[cluster_df['labels'] == 0].reset_index()
cluster_1 = cluster_df[cluster_df['labels'] == 1].reset_index()
cluster_2 = cluster_df[cluster_df['labels'] == 2].reset_index()
cluster_3 = cluster_df[cluster_df['labels'] == 3].reset_index()
cluster_4 = cluster_df[cluster_df['labels'] == 4].reset_index()


# ### Chart 15

# In[514]:


list_of_cluster=[cluster_0,cluster_1,cluster_2,cluster_3,cluster_4]


# In[515]:


# Create a scatter plot of the clusters with annotations for top cuisines
plt.figure(figsize=(11,6))
sns.scatterplot(x='Cost', y='Rating', hue='labels', data=cluster_df)

# Add annotations for top cuisines in each cluster
for i, df in enumerate(list_of_cluster):
    top_cuisines = df.drop(['index', 'Name', 'Cost', 'Rating', 'labels'], axis=1).sum().sort_values(ascending=False)[:3]
    top_cuisines_str = '\n'.join([f'{cuisine}: {count}' for cuisine, count in top_cuisines.items()])
    plt.annotate(f'Top cuisines in cluster {i}\n{top_cuisines_str}', 
                 xy=(df['Cost'].mean(), df['Rating'].mean()), 
                 ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
plt.xlabel('Cost')
plt.ylabel('Rating')
plt.title('Clustering of Restaurants')
plt.show()


# For each cluster, the top three cuisines are identified and annotated on the plot. The annotation includes the name of the cluster, its centroid location (mean cost and mean rating), and the top three cuisines and their counts within the cluster. This plot can be used to visually identify how the restaurants are grouped and the dominant features of each cluster.

# In[516]:


# Top cuisines in each cluster
for i,df in enumerate(list_of_cluster):
  print(f'Top cuisines in cluster {i}\n', df.drop(['index','Name','Cost','Rating','labels'],axis=1).sum(axis=0).sort_values(ascending=False)[:3],'\n')
     


# # 11. Conclusion

# The project was successful in achieving the goals of clustering and sentiment analysis. The clustering part provided insights into the grouping of restaurants based on their features, which can help in decision making for users and businesses. The sentiment analysis part provided insights into the sentiments expressed by the users in their reviews, which can help businesses in improving their services and user experience.
# 
# There are several potential areas for future work, such as implementing more advanced clustering algorithms and sentiment analysis techniques, incorporating more features such as images and menus of the restaurants, and exploring the relationships between the clustering and sentiment analysis results.

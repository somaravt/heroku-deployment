#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analsysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Read Data set

# In[2]:


df_review = pd.read_csv('sample30.csv')
print('Shape of dataset- ',df_review.shape)
df_review.head(5)


# In[3]:


# % of null values
null_values_per_column = (100*round((df_review.isnull().sum())/len(df_review),2)).sort_values(ascending=False)
print("% of null values per column:")
print(null_values_per_column)


# ### Here we see reviews_userProvince and reviews_userCity has ,more than 90% missing values and is not useful for our analysis, so removing these two columns

# In[4]:


df_review = df_review.drop('reviews_userProvince',axis=1)
df_review = df_review.drop('reviews_userCity',axis=1)


# In[5]:


df_review['reviews_doRecommend'].value_counts()


# ### Here we also see. reviews_doRecommend has 9% of null valus, we shall fill this false

# In[6]:


df_review.reviews_doRecommend = df_review['reviews_doRecommend'].fillna(False)
df_review.head()


# ### As we are recommending products based on review sentiment, we dont require following columns, so we'll drop it
# ##### 1. Id
# ##### 2. brand
# ##### 3. categories
# ##### 4. manufacturer
# ##### 5. reviews_date
# ##### 6. reviews_title

# In[7]:


df_review.drop(['id','brand','categories','manufacturer','reviews_date','reviews_title'],axis=1,inplace=True)
df_review.head()


# ### We will also drop reviews_didPurchase as it has half of the values as null

# In[8]:


df_review = df_review.drop('reviews_didPurchase',axis=1)


# In[9]:


df_review.reviews_rating.value_counts()


# #### Graph showing users rating on all products

# In[10]:


# Neighborhood also effects the price of the house and now we'll see the no. of sales in different neighboorhood
plt.figure(figsize=(12,7))
plt.hist(df_review.reviews_rating,bins=10)
plt.xticks(rotation=90)
plt.xlabel('ratings',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.show() 


# ### We have 20000+ 5 - ratings across the products

# In[11]:


Reviews_text = df_review.reviews_text
Reviews_text


# #### We'll process the text data

# In[12]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import string
import re
from nltk.stem import PorterStemmer
from textblob import TextBlob


# In[13]:


# Removing punctations in reviews
df_review.reviews_text = df_review.reviews_text.apply(lambda x: x.translate(x.maketrans('', '', string.punctuation)))


# In[14]:


# Removing Special Characters in reviews
df_review.reviews_text = df_review.reviews_text.apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))


# In[15]:


# Removing Emojis
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
df_review.reviews_text = df_review.reviews_text.apply(lambda x: emoji_pattern.sub(r'', x))


# In[16]:


# converting reviews to lower case
df_review.reviews_text = df_review.reviews_text.apply(lambda x: x.lower())


# In[17]:


# Removing stop words
df_review.reviews_text = df_review.reviews_text.apply(lambda x: " ".join([word for word in nltk.tokenize.word_tokenize(x) if word not in stopwords.words('english')]))


# In[18]:


# Performing stemming on the review text
ps = PorterStemmer()
df_review.reviews_text = df_review.reviews_text.apply(lambda x: " ".join([ps.stem(word) for word in nltk.tokenize.word_tokenize(x)]))


# In[19]:


df_review.reviews_text


# In[20]:


# Performing spelling corrections
df_review.reviews_text = df_review.reviews_text.apply(lambda x: TextBlob(x).correct())


# In[21]:


df_review.reviews_text = df_review.reviews_text.apply(lambda x: str(x))


# In[22]:


df_review


# In[23]:


# Separating features and response variables
X = df_review['reviews_text']
Y = df_review['user_sentiment']


# In[24]:


Y = Y.apply(lambda x: 1 if x == 'Positive' else 0)
Y.value_counts()


# In[25]:


# Spliting dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.30, random_state=44)


# In[26]:


y_train.value_counts()
X


# In[27]:


# Building Model
def buildModel(Model, Xtrain, Xtest):
    # Instantiate the model
    model = Model
    
    # Fitting model to the Training set (all features)
    model.fit(Xtrain, y_train)
    
    global y_pred
    # Predicting the Test set results
    y_pred = model.predict(Xtest)
    return model


# ## We have imbalanced data in the target variable, so we use class_weight as balanced in our models

# #### In order to help our model focus more on meaningful words, we can use a TF-IDF score (Term Frequency, Inverse Document Frequency). TF-IDF weighs words by how rare they are in our dataset, discounting words that are too frequent and just add to the noise.

# In[28]:


# Create the word vector with TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))
tfidf_vect_train = tfidf_vect.fit_transform(x_train)
tfidf_vect_train = tfidf_vect_train.toarray()
tfidf_vect_test = tfidf_vect.transform(x_test)
tfidf_vect_test = tfidf_vect_test.toarray()


# In[29]:


from xgboost import XGBClassifier

XGBoost = buildModel(XGBClassifier(class_weight='balanced'), tfidf_vect_train, tfidf_vect_test)

# Assign y_pred to a variable for further process
y_pred_tfidf_xgb = y_pred


# In[31]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


print('Confusion matrix: \n',confusion_matrix(y_test,y_pred_tfidf_xgb))
print(classification_report(y_test,y_pred))


# ## Recommendation System

# In[32]:


ratings = pd.read_csv('sample30.csv')
print('Shape of dataset- ',ratings.shape)


# In[33]:


ratings.head()


# In[34]:


# Dividing the dataset into trian and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings, test_size=0.30, random_state=44)


# In[35]:


print('train_shape - ',train.shape)
print('test_shape - ',test.shape)


# In[37]:


ratings.reviews_username = ratings.reviews_username.astype(str)


# In[38]:


ratings.reviews_username.size


# In[39]:


# Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user IDs.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(0)


# ### Creating dummy train & dummy test dataset
# These dataset will be used for prediction 
# - Dummy train will be used later for prediction of the products which has not been rated by the user. To ignore the products rated by the user, we will mark it as 0 during prediction. The products not rated by user is marked as 1 for prediction in dummy train dataset. 
# 
# - Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train.

# In[40]:


# Copy the train dataset into dummy_train
dummy_train = train.copy()
dummy_train.head()


# In[41]:


# The products not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[42]:


# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(1)
dummy_train.head()
dummy_train.shape


# # User Similarity Matrix

# ### Using Cosine Similarity

# In[43]:


from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[44]:


user_correlation.shape


# ### Using adjusted cosine

# #### Here, we are not removing the NaN values and calculating the mean only for the products rated by the user

# In[45]:


# Create a user-product matrix.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
)


# In[46]:


df_pivot.head()


# ### Normalising the rating of the products for each user around 0 mean

# In[47]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[48]:


df_subtracted.head()


# ### Now finding cosine similarity

# In[49]:


# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[50]:


user_correlation.shape


# ## Prediction User - User

# ##### Doing the prediction for the users which are positively related with other users, and not the users which are negatively related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for values less than 0. 

# In[51]:


user_correlation[user_correlation<0]=0
user_correlation


# ##### Rating predicted by the user (for products rated as well as not rated) is the weighted sum of correlation with the product rating (as present in the rating dataset). 

# In[52]:


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings


# In[53]:


user_predicted_ratings.shape


# ##### Since we are interested only in the products not rated by the user, we will ignore the products rated by the user by making it zero. 

# In[54]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# #### Finding top20 recommendations for user

# In[55]:


# Take the user ID as input.
user_input = str(input("Enter your user name:- "))
print(user_input)


# In[57]:


# d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:10]
d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
d


# In[88]:


import joblib

# Saving the model as a pickle in a file
joblib.dump(XGBoost,'XGBoost.pkl')


# In[89]:


def get_top_20_recommended_products(username):
    d = user_final_rating.loc[username].sort_values(ascending=False)[0:20]
    return d


# In[90]:


def get_reviews(product):
    return df_review[df_review['name']==product].reviews_text


# In[91]:


dict = {'product':[],
        'Sentiment':[]
       }
def get_top_5_products_using_sentiment_analysis(recommendations):
    predictions = pd.DataFrame(dict)
    for product in recommendations.index:
        reviews = get_reviews(product)
        transformed = tfidf_vect.transform(reviews)
        # predicting sentiment of reviews
        pred = XGBoost.predict(transformed)
        # calculating the +ve reviews percentage
        sentiment = 100*np.sum(pred)/pred.size
        predictions.loc[len(predictions.index)] = [product, sentiment]
        # Returning the top 5 +ve reviewed products
    return predictions.sort_values(['Sentiment'],ascending=False).head(5)['product']


# In[92]:


print('Top 20 Recommendations:-')
recommendations = get_top_20_recommended_products('joshua')
recommendations


# In[93]:


print('Top 5 recommendations using sentiment analysis:- ')
print(get_top_5_products_using_sentiment_analysis(recommendations))


# In[ ]:





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import ast 
import json
from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import PrecisionRecallDisplay, mean_squared_error, precision_recall_fscore_support, precision_recall_curve
from sklearn.pipeline import Pipeline

from wordcloud import WordCloud

from surprise import SVD, Reader, Dataset 
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from surprise import KNNWithMeans
from surprise import accuracy

from nltk import PorterStemmer


# This code defines a class called DatasetInfo that provides several methods for analyzing and processing datasets. 
# Let's go through the different methods:
class DatasetInfo:
    # This is the constructor method that initializes the class instance with a dataset.
    def __init__(self, dataset):
        self.dataset = dataset

    # This method checks if the dataset is a Pandas DataFrame and
    # returns the information about the dataset using the info() method of the DataFrame.
    def check_dataset_info(self):
        if isinstance(self.dataset, pd.DataFrame):
            return self.dataset.info()
        else:
            return "Invalid dataset type. Please provide a Pandas DataFrame."

    # This method checks if the dataset is either a NumPy array or a Pandas DataFrame and 
    # prints the shape of the dataset using the shape attribute.
    def check_dataset_shape(self):
        if isinstance(self.dataset, (np.ndarray, pd.DataFrame)):
            print("Dataset shape:", self.dataset.shape)
        else:
            print("Invalid dataset type. Please provide a NumPy array or a Pandas DataFrame.")

    #This method checks if the dataset is a Pandas DataFrame and 
    # returns the descriptive statistics of the dataset using the describe() method of the DataFrame.
    def get_dataset_statistics_describe(self):
        if isinstance(self.dataset, pd.DataFrame):
            return self.dataset.describe()
        else:
            return "Invalid dataset type. Please provide a Pandas DataFrame."

    # This is a static method that takes a string as input and converts it into a list of genres.
    # It assumes that the input string is in JSON format and extracts the genre names from the JSON objects.       
    @staticmethod
    def convert(self):
        result = []
        if isinstance(self, str):
            genres_list = json.loads(self)
            for genre in genres_list:
                result.append(genre['name'])
        return result
    
#This is another static method that takes an object as input and converts it into a list of names. It assumes that the input object is a string representation of a list of dictionaries.
#It iterates over the dictionaries and extracts the names, limiting the result to the first three names encountered.
    @staticmethod
    def convert3(obj):
        result = []
        count = 0
        for i in ast.literal_eval(obj):
            if count != 5:
                result.append(i['name'])
                count += 1
            else:
                break
        return result

# This is also a static method that takes a string as input and returns a list of directors' names. 
# It assumes that the input string is a string representation of a list of dictionaries. 
# It iterates over the dictionaries and extracts the names of directors based on the 'job' key being set to 'Director'.
    @staticmethod
    def get_directors(text):
        result = []
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                result.append(i['name'])
        return result
    
    @staticmethod
    def get_keywords(text):
        result = []
        for item in ast.literal_eval(text):
            result.append(item['name'])
        return result



def recommended_movies(title, cosine_sim, movies_data):
    
    indices = {title: index for index, title in enumerate(movies_data['title'])}
    
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores=sim_scores[1:11]
    
    # Get the movie indices
    ind=[]
    for (x,y) in sim_scores:
        ind.append(x)
        
    # Return the top 10 most similar movies
    tit=[]
    for x in ind:
        tit.append(movies_credits.iloc[x]['title'])
    return pd.Series(data=tit, index=ind)



def recommend_movies(title, cosine_sim, movies_data):
    # Create a dictionary to map movie titles to their indices
    indices = {title: index for index, title in enumerate(movies_data['title'])}

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    indices = [x for x, _ in sim_scores]

    # Return the top 10 most similar movies
    recommended_movies = movies_data.iloc[indices]['title']
    return recommended_movies



class DataFrameFiller:
    def __init__(self, df):
        self.df = df

    def fillna_random(self, columns):
        """Fill missing values in specified columns with random entries from the same columns."""
        for column in columns:
            # Get the non-null values from the column
            non_null_values = self.df.loc[self.df[column].notnull(), column]

            # Count the number of missing values in the column
            missing_count = self.df[column].isnull().sum()

            # Generate random values with the same length as missing values
            random_values = np.random.choice(non_null_values.values, size=missing_count, replace=True)

            # Assign the random values to the missing values in the column
            self.df.loc[self.df[column].isnull(), column] = random_values

        return self.df



def update_crew_with_director(data):
    # Create a new column called 'Directors' and assign the values of the 'crew' column to it
    data['Directors'] = data['crew']
    
    # Return the updated DataFrame
    return data



def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['Directors']) + ' ' + ' '.join(x['genres'])



# Calculate score for each qualified movie
def movie_score(x):
    v=x['vote_count']
    m=movies_credits['vote_count'].quantile(q=0.9)
    R=x['vote_average']
    C=movies_credits['vote_average'].mean()
    return ((R*v)/(v+m))+((C*m)/(v+m))




#Define a function to get movie recommendations for a user
def get_user_recommendations(user_Id, user_item_matrix, similarity_matrix, movies_credits, top_n=10):
    # Get the index of the user in the user-item matrix
    user_index = user_item_matrix.index.get_loc(user_Id)

    # Compute the similarity scores between the user and other users
    sim_scores = list(enumerate(similarity_matrix[user_index]))

    # Sort the users based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top N similar users
    top_user_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Get the movies that the top N similar users have rated highly
    top_movies = user_item_matrix.iloc[top_user_indices].sum(axis=0)
    top_movies = top_movies[top_movies == 0]  # Exclude movies already rated by the user

    # Get the movie details from the movie dataset
    recommendations = movies_credits.loc[top_movies.index]

    return recommendations




def recommended_movies(title, cosine_sim, movies_data):
    
    indices = {title: index for index, title in enumerate(movies_data['title'])}
    
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores=sim_scores[1:11]
    
    # Get the movie indices
    ind=[]
    for (x,y) in sim_scores:
        ind.append(x)
        
    # Return the top 10 most similar movies
    tit=[]
    for x in ind:
        tit.append(movies_credits.iloc[x]['title'])
    return pd.Series(data=tit, index=ind)


ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return" ".join(y)



def recommend(movie, new_movies, simliarity):
    movie_index = new_movies[new_movies["title"]==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:7]

    for i in movies_list:
        print(new_movies.iloc[i[0]].title)
        



# Function that takes in movie title as input and outputs most similar movies
def hybrid_recommendations(userId, title):
    
    indices = {title: index for index, title in enumerate(movies_data['title'])}
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores=sim_scores[1:11]
    
    # Get the movie indices
    ind=[]
    for (x,y) in sim_scores:
        ind.append(x)
        
    # Grab the title,movieid,vote_average and vote_count of the top 10 most similar movies
    tit=[]
    movieid=[]
    vote_average=[]
    vote_count=[]
    for x in ind:
        tit.append(movies_credits.iloc[x]['title'])
        movieid.append(movies_credits.iloc[x]['movieId'])
        vote_average.append(movies_credits.iloc[x]['vote_average'])
        vote_count.append(movies_credits.iloc[x]['vote_count'])

        
    # Predict the ratings a user might give to these top 10 most similar movies
    est_rating=[]
    for a in movieid:
        est_rating.append(svd.predict(userId, a, r_ui=None).est)  
        
    return pd.DataFrame({'index': ind, 'title':tit, 'movieId':movieid, 'vote_average':vote_average, 'vote_count':vote_count,'estimated_rating':est_rating}).set_index('index').sort_values(by='estimated_rating', ascending=False)


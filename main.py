import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

metadata = pd.read_csv('data/movies.csv', low_memory=False)

metadata['year'] = metadata['title'].apply(lambda x: x[-5:-1])
metadata['title'] = metadata['title'].apply(lambda x: x[:-7])
metadata['genres'] = metadata['genres'].apply(lambda x: x.replace('|',', '))

print metadata.head()

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['title'] = metadata['title'].fillna('')


# ################################################

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['title'])

#Output the shape of tfidf_matrix
print tfidf_matrix.shape

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# # Compute the cosine similarity matrix
# cosine_sim_l = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print cosine_sim

sim_scores = list(enumerate(cosine_sim[1-1]))
print sim_scores

# Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# idx = metadata[metadata['title'] == 'Titanic']
# print idx

# # #Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # idx = metadata[metadata['title'] == title]
    print 'idx'
    print idx

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # print 'sim_scores'
    # print sim_scores

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # print 'SORTED sim_scores'
    # print sim_scores

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    print '10 Most sim_scores'
    print sim_scores

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

# print get_recommendations('Dangerous Minds')


# ##################################################


# def clean_data(x):
#     if isinstance(x, list):
#         return [str.lower(i.replace(" ", "")) for i in x]
#     else:
#         #Check if director exists. If not, return empty string
#         if isinstance(x, str):
#             return str.lower(x.replace(" ", ""))
#         else:
#             return ''

# # Apply clean_data function to your features.
# features = ['genres','year']

# for feature in features:
#     metadata[feature] = metadata[feature].apply(clean_data)

# def create_soup(x):
#     return ' '.join(x['title']) + ' ' + ' '.join(x['genres']) + ' ' + x['year']


# # Create a new soup feature
# metadata['soup'] = metadata.apply(create_soup, axis=1)
# print metadata['soup'].head()


# # Import CountVectorizer and create the count matrix
# from sklearn.feature_extraction.text import CountVectorizer

# count = CountVectorizer(stop_words='english')
# count_matrix = count.fit_transform(metadata['soup'])

# # Compute the Cosine Similarity matrix based on the count_matrix
# from sklearn.metrics.pairwise import cosine_similarity

# cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# # Reset index of your main DataFrame and construct reverse mapping as before
# metadata = metadata.reset_index()
# indices = pd.Series(metadata.index, index=metadata['title'])


# print get_recommendations('Dangerous Minds', cosine_sim2)
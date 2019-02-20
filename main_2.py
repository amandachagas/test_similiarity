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


# Get the pairwsie similarity scores of all movies with that movie
sim_scores = list(enumerate(cosine_sim[1-1]))

# Sort the movies based on the similarity scores
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

 # Get the scores of the 10 most similar movies
sim_scores = sim_scores[1:11]
print '10 Most sim_scores'
print sim_scores

print metadata[2508]

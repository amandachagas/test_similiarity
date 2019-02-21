import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

metadata = pd.read_csv('data/movies.csv', low_memory=False)

metadata['year'] = metadata['title'].apply(lambda x: x[-5:-1])
metadata['title'] = metadata['title'].apply(lambda x: x[:-7])
metadata['genres'] = metadata['genres'].apply(lambda x: x.replace('|',', '))

# print metadata.head()

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['title'] = metadata['title'].fillna('')
metadata['genres'] = metadata['genres'].fillna('')


# ################################################

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['title'])
tfidf_matrix_genres = tfidf.fit_transform(metadata['genres'])

#Output the shape of tfidf_matrix
# print tfidf_matrix.shape

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# # Compute the cosine similarity matrix
# cosine_sim_l = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_2 = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)
# print cosine_sim
# print cosine_sim_2


# Get the pairwsie similarity scores of all movies with that movie
sim_scores = list(enumerate(cosine_sim[1-1]))
sim_scores_genres = list(enumerate(cosine_sim_2[1-1]))


total_sim_score = []

for i in range(len(sim_scores)):
	aux = (sim_scores[i][1]*0.6) + (sim_scores_genres[i][1]*0.4)
	total_sim_score.append((i, aux))

# Sort the movies based on the similarity scores
total_sim_score = sorted(total_sim_score, key=lambda x: x[1], reverse=True)

 # Get the scores of the 10 most similar movies
total_sim_score = total_sim_score[1:11]
print '10 Most total_sim_score'
print total_sim_score

df_total = pd.DataFrame(columns=['movieId', 'title', 'genres', 'year'])

for x in total_sim_score:
	df_total = df_total.append(metadata.iloc[x[0]], ignore_index=True)

print df_total.head(10)

# ### GET THE TITLES MOST SIMILAR AND GENRES
# # Sort the movies based on the similarity scores
# sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# sim_scores_genres = sorted(sim_scores_genres, key=lambda x: x[1], reverse=True)

#  # Get the scores of the 10 most similar movies
# sim_scores = sim_scores[1:11]
# sim_scores_genres = sim_scores_genres[1:11]
# print '10 Most sim_scores'
# print sim_scores

# # print metadata.iloc[2506]

# df_title = pd.DataFrame(columns=['movieId', 'title', 'genres', 'year'])

# for x in sim_scores:
# 	df_title = df_title.append(metadata.iloc[x[0]], ignore_index=True)

# print df_title.head(10)

# print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'

# print '10 Most sim_scores_genres'
# print sim_scores_genres

# # print metadata.iloc[2506]

# df_genres = pd.DataFrame(columns=['movieId', 'title', 'genres', 'year'])

# for x in sim_scores_genres:
# 	df_genres = df_genres.append(metadata.iloc[x[0]], ignore_index=True)

# print df_genres.head()

# ###  END GET THE TITLES MOST SIMILAR AND GENRES




# for x, y in sim_scores, sim_scores_genres:
# 	aux = ((x[1]*6)+(y[1]*4))/10
# 	print aux

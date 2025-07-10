import numpy as np
import pandas as pd
import difflib #for matching with variations due to user mistake
from sklearn.feature_extraction.text import TfidfVectorizer #vector conversion-text to numericals
from sklearn.metrics.pairwise import cosine_similarity #gives similarity score

movie_data=pd.read_csv("C:\\Users\\Vaishnavi A Das\\Desktop\\B.E\\projects\\movies.csv") #dont forget double slash mone!
#print(movie_data.head())

chars= ['genres','keywords','tagline','cast','director']
combined_data = ''
for char in chars:
    combined_data += movie_data[char].fillna('') + ' '
#print(combined_data)

vectorize=TfidfVectorizer()
char_vectors=vectorize.fit_transform(combined_data)
#print(char_vectors) #covert text to vectors

similarity=cosine_similarity(char_vectors)
#print(similarity.shape)

movie_name=input('enter movie name:')
list_of_all_titles=movie_data['title'].tolist()

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
#print(find_close_match)

close_match=find_close_match[0]
#print(close_match)

index_of_movie=movie_data[movie_data.title==close_match]['index'].values[0]
#print(index_of_movie)

similarity_score=list(enumerate(similarity[index_of_movie]))
#print(similarity)

sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1] ,reverse=True)
#print(sorted_similar_movies)

i=0
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movie_data[movie_data.index==movie[0]]['title'].values[0]
    if i<30:
        print(i,title_from_index)
        i+=1
        
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("Movie Review App")
st.subheader("Check whether your comment or review about a movie is a positive or negative")

st.write(" Number of Classes : 2")
st.write(" Classifier : Naive Bayes")
st.write("Accuracy = 0.86")

user_input = st.text_input("Your Review:")

if user_input != '':
  movie_review_data = pd.read_table('/content/drive/My Drive/Smartknowers major project/SmartKnowers major project - movie review sentiment analysis/movie review.tsv')
  movie_review_data.drop(['fold_id', 'cv_tag', 'html_id', 'sent_id'], axis=1, inplace=True)
  movie_review_data = movie_review_data.sample(frac=1)

  x = movie_review_data.iloc[:,0].values
  y = movie_review_data.iloc[:,1].values
  review_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
  review_model.fit(x,y)


  y_pred = review_model.predict([user_input])

  if y_pred[0] == "neg":
    st.write("Prediction:")
    st.write("Your review about the movie is a Negative review")
  else:
    st.write("Prediction:")
    st.write("Your review about the movie is a Positive review")
 

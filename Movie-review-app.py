{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Movie-review-app.py",
      "provenance": [],
      "collapsed_sections": [],
      "mount_file_id": "15hlIqd1PH_OQKlpSoZVlARvPR7ir8DCN",
      "authorship_tag": "ABX9TyOuav+uSS+FjykaRemjfsVF",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AndrewPrasaath/Movie-Review-Sentiment-analysis/blob/master/Movie-review-app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Jw7WVZTyiC1x",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "\n",
        "st.title(\"Movie Review App\")\n",
        "st.subheader(\"Check whether your comment or review about a movie is a positive or negative\")\n",
        "\n",
        "st.write(\" Number of Classes : 2\")\n",
        "st.write(\" Classifier : Naive Bayes\")\n",
        "st.write(\"Accuracy = 0.86\")\n",
        "\n",
        "user_input = st.text_input(\"Your Review:\")\n",
        "\n",
        "if user_input != '':\n",
        "  movie_review_data = pd.read_table('/content/drive/My Drive/Smartknowers major project/SmartKnowers major project - movie review sentiment analysis/movie review.tsv')\n",
        "  movie_review_data.drop(['fold_id', 'cv_tag', 'html_id', 'sent_id'], axis=1, inplace=True)\n",
        "  movie_review_data = movie_review_data.sample(frac=1)\n",
        "\n",
        "  x = movie_review_data.iloc[:,0].values\n",
        "  y = movie_review_data.iloc[:,1].values\n",
        "  review_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])\n",
        "  review_model.fit(x,y)\n",
        "\n",
        "\n",
        "  y_pred = review_model.predict([user_input])\n",
        "\n",
        "  if y_pred[0] == \"neg\":\n",
        "    st.write(\"Prediction:\")\n",
        "    st.write(\"Your review about the movie is a Negative review\")\n",
        "  else:\n",
        "    st.write(\"Prediction:\")\n",
        "    st.write(\"Your review about the movie is a Positive review\")"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
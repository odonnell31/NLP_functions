# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 09:19:34 2022

@author: Michael ODonnell
"""

# grab some text to run functions on
# request the raw text of The Great Gatsby
# you will need to leverage the requests package
import requests
r = requests.get(r'https://www.gutenberg.org/cache/epub/64317/pg64317.txt')
great_gatsby = r.text

# first, remove unwanted new line and tab characters from the text
for char in ["\n", "\r", "\d", "\t"]:
    great_gatsby = great_gatsby.replace(char, " ")
    
# you can also subset for the book text
# (removing the project gutenburg introduction/footnotes)
great_gatsby = great_gatsby[1433:277912]

# tokenize your string with nltk
def tokenize_text(text: str):
    
    # import needed packages
    import nltk
    import re
    
    # remove unwanted new line and tab characters from the text
    for char in ["\n", "\r", "\d", "\t"]:
        text = text.replace(char, " ")
    
    # lowercase the text
    text = text.lower()
    
    # remove punctuation from text
    text = re.sub(r"[^\w\s]", "", text)
    
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # remove stopwords from txt_tokens and word_tokens
    from nltk.corpus import stopwords
    english_stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in english_stop_words]
    
    # return your tokens
    return tokens

#tokens = tokenize_text(text = great_gatsby)
#print(tokens)
  
# lemmatize your tokens with nltk
def lemmatize_tokens(tokens):
    
    # import needed packages
    import nltk
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # return your lemmatized tokens
    return lemmatized_tokens

#lemmatized_tokens = lemmatize_tokens(tokens = tokens)
#print(lemmatized_tokens)

# return the most common tokens
def return_top_tokens(tokens,
                      top_N = 10):

    # first, count the frequency of every unique token
    word_token_distribution = nltk.FreqDist(tokens)
    
    # next, filter for only the most common top_N tokens
    # also, put this in a dataframe
    top_tokens = pd.DataFrame(word_token_distribution.most_common(top_N),
                              columns=['Word', 'Frequency'])
    
    # return the top_tokens dataframe
    return top_tokens

## run the return_top_tokens and print the results
#top_tokens = return_top_tokens(tokens = lemmatized_tokens,
#                               top_N = 10)
#print(top_tokens)
    
# return the most common bi-grams
from nltk.collocations import BigramCollocationFinder

def return_top_bigrams(tokens,
                       top_N = 10):
    
    # collect bigrams
    bcf = BigramCollocationFinder.from_words(tokens)
    
    # put bigrams into a dataframe
    bigram_df = pd.DataFrame(data = bcf.ngram_fd.items(),
                             columns = ['Bigram', 'Frequency'])
    
    # sort the dataframe by frequency
    bigram_df = bigram_df.sort_values(by=['Frequency'],ascending = False).reset_index(drop=True)
    
    # filter for only top bigrams
    bigram_df = bigram_df[0:top_N]
    
    # return the bigram dataframe
    return bigram_df

## run the return_top_bigrams function and print the results
#bigram_df = return_top_bigrams(tokens = lemmatized_tokens,
#                               top_N = 10)
#print(bigram_df)
    
# Let's try tri-grams (sets of 3 words)
from nltk.collocations import TrigramCollocationFinder

def return_top_trigrams(tokens,
                        top_N = 10):
    
    # collect bigrams
    tcf = TrigramCollocationFinder.from_words(tokens)

    # put trigrams into a dataframe
    trigram_df = pd.DataFrame(data = tcf.ngram_fd.items(),
                              columns = ['Trigram', 'Frequency'])
    
    # sort the dataframe by frequency
    trigram_df = trigram_df.sort_values(by = ['Frequency'], ascending = False).reset_index(drop=True)
    
    # filter for only top trigrams
    trigram_df = trigram_df[0:top_N]
    
    return trigram_df

## run the return_top_trigrams function and print the results
#trigram_df = return_top_trigrams(tokens = lemmatized_tokens,
#                               top_N = 10)
#print(trigram_df)

# create a wordcloud
def flexbile_wordcloud_function(text: str,
                                output_filepath: str,
                                mask_path = None,
                                white_mask_background = True,
                                width = 725,
                                height = 300,
                                background_color = "white",
                                colormap = "viridis",
                                contour_color = "steelblue",
                                contour_width = 3,
                                collocations = False,
                                max_words = 2000,
                                max_font_size = 40,
                                min_font_size = 4,
                                prefer_horizontal = 0.9,
                                include_numbers = True):
    
    # start function timer
    import time
    start = time.time()
    
    # tokenize and lemmatize your text
    tokens = tokenize_text(text = text)
    lemmatized_tokens = lemmatize_tokens(tokens = tokens)
    
    # import needed packages
    from wordcloud import WordCloud
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    # create a wordcloud object without a mask
    if mask_path == None:
    
        # create a WordCloud object
        wordcloud = WordCloud(width = width,
                              height = height,
                              background_color = background_color,
                              colormap = colormap,
                              collocations = collocations,
                              max_words = max_words,
                              max_font_size = max_font_size,
                              min_font_size = min_font_size,
                              prefer_horizontal = prefer_horizontal,
                              include_numbers = include_numbers)
    
    # create a wordcloud object with a mask image
    elif mask_path != None:
        
        # open the mask image as a numpy array
        mask = np.array(Image.open(mask_path))
        
        # if your mask has a black background update to white
        if white_mask_background == False:
            mask[mask[:, :] == 0] = 255
        
        # create a WordCloud object
        wordcloud = WordCloud(mask = mask,
                              width=mask.shape[1],
                              height=mask.shape[0],
                              background_color = background_color,
                              colormap = colormap,
                              contour_color = contour_color,
                              contour_width = contour_width,
                              collocations = collocations,
                              max_words = max_words,
                              max_font_size = max_font_size,
                              min_font_size = min_font_size,
                              prefer_horizontal = prefer_horizontal,
                              include_numbers = include_numbers)

    # generate a word cloud (must join the tokens into a string)
    wordcloud.generate(','.join(lemmatized_tokens))

    # end wordcloud timer
    end = time.time()
    print(f"wordcloud created in {round(end-start, 1)} seconds")
    
    # print, save, and return the wordcloud
    plt.imshow(wordcloud)
    wordcloud.to_file(output_filepath)
    return wordcloud.to_image()

# Ok, last point - What about if your text is stored in a dataframe column?
# Well, one way to start (the simplest) is to pull all of the columns text into a single string
# create remarks dataframe from self.response_report_df
def tokenize_dataframe_columns(dataframe,
                               column):
    
    # create string from columnremarks text
    string_from_column = dataframe[column].cat(sep=' ')
    
    # tokenize the new string using the function we already defined above
    tokens = tokenize_text(text = string_from_column)
    
    # return tokens
    return tokens

# print out some information about the text
def text_summary_info(text: str,
                      characters: list):
    # what's the data type of your text
    print(f"the type of your data: {type(text)}")
    
    # how long is your text (in characters)?
    print(f"length = {len(text)} characters")
    
    # Which of your favorite characters is most mentioned?
    
    # create an empty dict to keep track of mentions by character
    reference_dict = {}
    # loop through each character to count their mentions
    for character in characters:
        reference_dict[character] = text.lower().count(character)
    # turn your dictionary into a pandas dataframe and print it
    import pandas as pd
    df = pd.DataFrame(list(reference_dict.items()),
                     columns = ["character", "mentions"])
    df = df.set_index("character")
    df = df.sort_values(by = "mentions",
                       ascending = False)
    print("\n")
    print(df)

#text_summary_info(text = great_gatsby,
#                  characters = ["daisy", "jay", "nick", "tom", "myrtle"])
    
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

def return_sentiment_df(tokens):

    # initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # create some counters for sentiment of each token
    positive_tokens = 0
    negative_tokens = 0
    neutral_tokens = 0
    compound_scores = []
        
    # loop through each token
    for token in tokens:
        
        if sia.polarity_scores(token)["compound"] > 0:
            
            positive_tokens += 1
            compound_scores.append(sia.polarity_scores(token)["compound"])
            
        elif sia.polarity_scores(token)["compound"] < 0:
            
            negative_tokens += 1
            compound_scores.append(sia.polarity_scores(token)["compound"])
              
        elif sia.polarity_scores(token)["compound"] == 0:
            
            neutral_tokens += 1
            compound_scores.append(sia.polarity_scores(token)["compound"])
            
    # put sentiment results into a dataframe
    compound_score_numbers = [num for num in compound_scores if num != 0]
    sentiment_df = pd.DataFrame(data = {"total_tokens" : len(tokens),
                                        "positive_tokens" : positive_tokens,
                                        "negative_tokens" : negative_tokens,
                                        "neutral_tokens" : neutral_tokens,
                                        "compound_sentiment_score" : sum(compound_score_numbers) / len(compound_score_numbers)},
                                index = [0])

    # return sentiment_df
    return sentiment_df

#sentiment_df = return_sentiment_df(tokens = lemmatized_tokens)
#print(sentiment_df)

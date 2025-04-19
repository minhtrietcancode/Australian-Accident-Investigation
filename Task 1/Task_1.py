import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    '''
        - Parameter: text is a str() 
        - Return: a str() which is text being converted to lowercase
                  and removing punctuation
    '''
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    return text

def Task_1():
    # Set of English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Read the CSV data
    df = pd.read_csv('../dataset/accident.csv')
    
    # Extract the serie pf description from dataframe and 
    # then remove all of the entry with NaN value, also make sure everything will be string
    descriptions = df['DCA_DESC'].dropna().astype(str)

    # Extract and process all of the description from the serie 'descriptions'
    processed_texts = [preprocess_text(desc) for desc in descriptions]
    
    # Tokenize and remove stop words --> combine everything into a single list only 
    all_words = []
    for text in processed_texts:
        tokens = word_tokenize(text)
        filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]
        all_words.extend(filtered_words)
    
    # Apply Bag of Words (counting frequency)
    word_counts = Counter(all_words)
    
    # Get top 20 words
    top_20 = dict(word_counts.most_common(20))
    
    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color='white')
    wordcloud_image = wc.generate_from_frequencies(top_20)
    
    # Plot and save
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_image, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig("Task_1_word_cloud.png")

# and now run it 
Task_1()
    
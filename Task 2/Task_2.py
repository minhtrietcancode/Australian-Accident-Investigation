import pandas as pd
import matplotlib.pyplot as plt
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''
NOTE THAT I HAVE ALREADY VERIFIED BEFORE OUTSIDE THIS FUNCTION THAT ALL OF THE INPUT 
FOR THE COLUMN ACCIDENT_TIME ARE ALL IN THE FORMAT "HH:MM:SS"

if you want to check then try this code 
import pandas as pd
import re

# Load the dataset
df = pd.read_csv('accident.csv')

# Extract the ACCIDENT_TIME column
time = df['ACCIDENT_TIME']

# Function to print invalid time values
def print_invalid_times(time_series):
    pattern = r'^\d{2}:\d{2}:\d{2}$'  # Matches "HH:MM:SS" format
    invalid_times = time_series[~time_series.astype(str).str.match(pattern, na=False)]

    print("Invalid time values:")
    print(invalid_times.to_list())

# Call the function
print_invalid_times(time)
'''


def get_time_of_day(accident_time):
    """
    Convert an accident time string in 'HH:MM:SS' format into a time-of-day category.
    Returns "Unknown" if the input is not in the expected format or has an invalid hour.
    """
    try:
        # Extract the hour part safely
        hour = int(accident_time[:2])  # First two characters represent the hour
    except ValueError:
        return "Unknown"

    # Ensure the hour is within a valid range
    if 0 <= hour < 6:
        return "Late Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Unknown"  # Handle cases where hour is out of range


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


def Task_2():
    """
    Processes the accident dataset by:
     1. Adding a TIME_OF_DAY column based on ACCIDENT_TIME.
     2. Plotting a bar chart for accident counts per TIME_OF_DAY (saved as task2_2_timeofday.png).
     3. Generating a 2x2 grid of pie charts showing the top 10 words in DCA_DESC for each TIME_OF_DAY
        (saved as task2_2_wordpies.png) with legends for each pie.
     4. Plotting a stacked bar chart for Mondays, Fridays and Sundays comparing accident counts in each TIME_OF_DAY
        (saved as task2_2_stackbar.png) using a custom light-to-dark colour scheme from Morning to Late Night.
    """
    # Read the dataset
    df = pd.read_csv('../dataset/accident.csv')

    # Set of English stopwords
    stop_words = set(stopwords.words('english'))

    # Create TIME_OF_DAY column based on ACCIDENT_TIME
    df['TIME_OF_DAY'] = df['ACCIDENT_TIME'].astype(str).apply(get_time_of_day)

    # Filter out any "Unknown" time categories if they exist
    df_valid = df[df['TIME_OF_DAY'] != "Unknown"]

    '''
    REQUIREMENT 1: Plot 1: Bar Chart comparing accidents by TIME_OF_DAY
    '''
    # Ensure the ordering of categories
    categories = ["Morning", "Afternoon", "Evening", "Late Night"]

    # Filter to count the value for each time-of-day period
    accident_counts = df_valid['TIME_OF_DAY'].value_counts().reindex(categories, fill_value=0)

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    accident_counts.plot(kind='bar', color='skyblue')
    plt.title("Accidents by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Task_2_timeofday.png")
    plt.close()

    '''
    REQUIREMENT 2: Plot 2: Pie Charts for Top 10 Words in DCA_DESC per TIME_OF_DAY
    '''
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for idx, category in enumerate(categories):
        # Filter for the current TIME_OF_DAY category
        subset = df_valid[df_valid['TIME_OF_DAY'] == category]

        # Get the DCA_DESC column for each category and process
        descriptions = subset['DCA_DESC'].dropna().astype(str)

        # Process text and tokenize
        all_words = []
        for desc in descriptions:
            processed_text = preprocess_text(desc)
            tokens = word_tokenize(processed_text)
            filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]
            all_words.extend(filtered_words)

        # Count frequencies and get top 10 words
        word_counter = Counter(all_words)
        top10 = word_counter.most_common(10)

        if top10:
            labels, sizes = zip(*top10)
            # Draw the pie chart without direct labels
            wedges, texts, autotexts = axs[idx].pie(sizes, autopct='%1.1f%%', startangle=140)
            # Add a legend using the wedges and labels
            axs[idx].legend(wedges, labels, title="Top 10 Words", loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            axs[idx].text(0.5, 0.5, "No data available", ha='center', va='center')

        axs[idx].set_title(f"{category}")

    plt.suptitle("Top 10 Frequent Words in DCA_DESC per Time of Day", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Task_2_wordpies.png")
    plt.close()

    '''
    REQUIREMENT 3: Plot 3: Stacked Bar Chart for Mondays, Fridays, and Sundays using DAY_WEEK_DESC
    '''
    days_of_interest = ["Monday", "Friday", "Sunday"]
    df_days = df_valid[df_valid['DAY_WEEK_DESC'].isin(days_of_interest)]

    # Group by DAY_WEEK_DESC and TIME_OF_DAY and count the accidents
    pivot = df_days.groupby(['DAY_WEEK_DESC', 'TIME_OF_DAY']).size().unstack(fill_value=0)

    # Ensure a consistent order for TIME_OF_DAY columns
    time_order = ["Morning", "Afternoon", "Evening", "Late Night"]
    pivot = pivot.reindex(columns=time_order, fill_value=0)

    # Also, order the days as: Monday, Friday, Sunday
    pivot = pivot.reindex(days_of_interest)

    # Define custom colours: light colour for Morning and a dark colour for Late Night.
    custom_colors = {
        "Morning": "#a6cee3",  # light blue
        "Afternoon": "#1f78b4",  # moderate blue
        "Evening": "#08519c",  # darker blue
        "Late Night": "#08306b"  # darkest blue
    }

    pivot.plot(kind='bar', stacked=True, figsize=(8, 6),
               color=[custom_colors[tod] for tod in time_order])
    plt.title("Accident Counts by Time of Day on Monday, Friday, and Sunday")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("Task_2_stackbar.png")
    plt.close()

# run the code above
Task_2()

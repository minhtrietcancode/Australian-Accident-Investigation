'''

- TEMPORAL PATTERN ANALYSIS
For task 4, we will focus on these things:
    + Create a new column MONTH by extracting the month from ACCIDENT_DATE
    + Generate a line plot showing the monthly distribution of accidents
      --> save as Task_4_monthly.png
    + Create a heatmap visualization showing accident frequency by day of week and time of day
      --> save as Task_4_weekday_time_heatmap.png
    + Identify seasonal patterns in accident types using appropriate visualizations
      --> save as Task_4_seasonal.png

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset into a dataframe
df = pd.read_csv('../dataset/accident.csv')

'''
    REQUIREMENT 1: Generate a line plot to see the monthly distribution of accidents
                   by extracting MONTH from ACCIDENT_DATE column 
'''

# extract the column ACCIDENT_DATE
date = df['ACCIDENT_DATE']

def extract_month(date):
    '''
    - Function receive a date as a string of the form "YYYY-MM-DD"
    - Return an integer by extracting the month from the string
    '''
    output = int(date[5:7])
    return output if 1 <= output <= 12 else 0

# Note that i have already verified outside that all the data entry of the date column are all in the expected form
df['MONTH'] = df['ACCIDENT_DATE'].astype(str).apply(extract_month)

# oke all good all data entry is valid, now need to clean
# month = df['MONTH']
# print(month.value_counts(dropna=False))

# And now group data by MONTH to prepare for plotting the distribution
distribution_by_month_df = df.groupby(['MONTH']).size().reset_index(name='count')

# Line plot for accident distribution by month
plt.figure(figsize=(10, 6))
plt.plot(distribution_by_month_df['MONTH'], distribution_by_month_df['count'], marker='o', linestyle='-', color='blue')

plt.title('Accident Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.xticks(range(1, 13))  # Make sure X-axis has 1 to 12 for months
plt.grid(True)

# Save the figure if needed
plt.tight_layout()
plt.savefig('Task_4_monthly.png', dpi=300)

'''
    REQUIREMENT 2: Generate a heatmap visualization showing accident frequency by day of week and time of day
                   --> save as Task_4_weekday_time_heatmap.png
'''
# Function to convert ACCIDENT_TIME to four categories: Morning, Afternoon, Evening, Late Night
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

# Create TIME_OF_DAY column based on ACCIDENT_TIME
df['TIME_OF_DAY'] = df['ACCIDENT_TIME'].astype(str).apply(get_time_of_day)

# Group by TIME_OF_DAY and DAY_WEEK_DESC, count occurrences
grouped = df.groupby(['DAY_WEEK_DESC', 'TIME_OF_DAY']).size().reset_index(name='count')

# Pivot the table for heatmap: rows = DAY_WEEK_DESC, columns = TIME_OF_DAY
heatmap_data = grouped.pivot(index='DAY_WEEK_DESC', columns='TIME_OF_DAY', values='count').fillna(0)

# Optional: define a custom order if you want days/time to look more natural
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
time_order = ['Morning', 'Afternoon', 'Evening', 'Late Night']

# Reindex for consistent order
heatmap_data = heatmap_data.reindex(index=day_order, columns=time_order)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu')

plt.title('Accident Frequency by Day of Week and Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Day of Week')

# Save the figure
plt.tight_layout()
plt.savefig('Task_4_weekday_time_heatmap.png', dpi=300)


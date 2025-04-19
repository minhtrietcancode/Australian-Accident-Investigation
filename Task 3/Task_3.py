# - SEVERITY ANALYSIS
# We will focus on these points for task 3:
#     + Generate a heat map showing the correlation between SEVERITY, NO_OF_VEHICLES, and SPEED_ZONE
#       --> save as Task_3_heatmap.png
#     + Create a stacked bar chart comparing severity levels across different LIGHT_CONDITION categories
#       --> save as Task_3_lightcondition.png
#     + Analyze the relationship between SEVERITY and NO_PERSONS_KILLED / NO_PERSON_INJ_2 / NO_PERSON_INJ_3
#       --> save as Task_3_injuries.png

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a df for future processing
accident = pd.read_csv('../dataset/accident.csv')

'''
    REQUIREMENT 1: WORKING WITH THE CORRELATION BETWEEN SEVERITY, NO_OF_VEHICLES, and SPEED_ZONE
'''
# Firstly remove rows with NaN values at the columns that we are looking for
target_columns = ['SEVERITY', 'NO_OF_VEHICLES', 'SPEED_ZONE']
first_cleaned_accident = accident.dropna(subset = target_columns)

'''
- This block of code is just to check is there any entry data of these columns that are not integer, but it is all
  good though 
severity = first_cleaned_accident['SEVERITY']
vehicle_num = first_cleaned_accident['NO_OF_VEHICLES']
speed_zone = first_cleaned_accident['SPEED_ZONE']

checking_column = [severity, vehicle_num, speed_zone]
for check in checking_column:
    print(check.value_counts(dropna=False))
'''

# Now create a new dataframe of just these 3 targeted columns
new_accident = first_cleaned_accident[['SEVERITY', 'NO_OF_VEHICLES', 'SPEED_ZONE']]

# Set figure size (width, height in inches)
plt.figure(figsize=(10, 8))

# now calculate the correlation matrix
relation_matrix = new_accident.corr()

# and then plot the heatmap for the new generated dataframe of just these 3 targeted dataframe
sns.heatmap(relation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between severity levels, number of vehicles involved, and speed zone in these accidents')

# Save the plot
plt.savefig('Task_3_heatmap.png', dpi=300, bbox_inches='tight')  # high quality + trims whitespace

'''
    REQUIREMENT 2: Comparing severity level between different LIGHT_CONDITION categories 
'''
# Firstly remove rows with NaN values at the columns of this requirement
second_target_col = ['SEVERITY', 'LIGHT_CONDITION']
second_cleaned_accident = accident.dropna(subset = second_target_col)

'''
- This piece of code is just to verify that all of the data entry of these 2 columns are in corerct format 
severity = second_cleaned_accident['SEVERITY']
light_condition = second_cleaned_accident['LIGHT_CONDITION']

checking_col = [severity, light_condition]
for check in checking_col:
    print(check.value_counts(dropna=False))
'''

# Step 2: Group and pivot
grouped = second_cleaned_accident.groupby(['LIGHT_CONDITION', 'SEVERITY']).size().reset_index(name='count')
pivot_table = grouped.pivot(index='LIGHT_CONDITION', columns='SEVERITY', values='count').fillna(0)

# Ensure severity levels are in order (1 to 4)
pivot_table = pivot_table[[1, 2, 3, 4]] if set([1,2,3,4]).issubset(pivot_table.columns) else pivot_table

# Step 3: Define custom colors (lighter to darker)
colors = ['#c6dbef', '#6baed6', '#3182bd', '#08519c']  # light blue → dark blue

# Step 4: Plot stacked bar chart
pivot_table.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)

plt.title('Stacked Bar Chart of Severity Levels by Light Condition')
plt.xlabel('Light Condition')
plt.ylabel('Number of Accidents')
plt.legend(title='Severity Level')
plt.xticks(rotation=0)

# Step 5: Save it
plt.tight_layout()
plt.savefig('Task_3_lightcondition.png', dpi=300)

'''
    REQUIREMENT 3: Analyze the relationship between SEVERITY level and the number of victims of different types
'''
# Firstly remove row with NaN values at our targeted columns
third_target = ['SEVERITY', 'NO_PERSONS_KILLED', 'NO_PERSONS_INJ_2', 'NO_PERSONS_INJ_3']
third_accident = accident.dropna(subset=third_target)

'''
- This piece of code is just to check the values of each columns 
severity = third_accident['SEVERITY']
kill = third_accident['NO_PERSONS_KILLED']
two = third_accident['NO_PERSONS_INJ_2']
three = third_accident['NO_PERSONS_INJ_3']

checking = [severity, kill, two, three]
for check in checking:
    print(check.value_counts(dropna=False))
    print('\n')
'''

# Step 2: Group by severity and sum up the number of victims
grouped_victims = third_accident.groupby('SEVERITY')[['NO_PERSONS_KILLED', 'NO_PERSONS_INJ_2', 'NO_PERSONS_INJ_3']].sum()

# Step 3: Plot stacked bar chart
colors = ['#fb6a4a', '#fcae91', '#fee5d9']  # red for killed, then lighter for injured

grouped_victims.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)

plt.title('Victim Types by Severity Level')
plt.xlabel('Severity Level')
plt.ylabel('Number of Victims')
plt.legend(['Killed', 'Serious Injuries', 'Minor Injuries'])
plt.xticks(rotation=0)

# Save the figure
plt.tight_layout()
plt.savefig('Task_3_injuries.png', dpi=300)

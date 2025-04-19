# - SEVERITY ANALYSIS
# We will focus on these points for task 3:
#     + Generate a heat map showing the correlation between SEVERITY, NO_OF_VEHICLES, and SPEED_ZONE
#       --> save as Task_3_heatmap.png
#     + Create a grouped bar chart comparing severity levels across different LIGHT_CONDITION categories
#       --> save as Task_3_lightcondition.png
#     + Analyze the relationship between SEVERITY and NO_PERSONS_KILLED / NO_PERSON_INJ_2 / NO_PERSON_INJ_3
#       --> save as Task_3_injuries.png

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a df for future processing
accident = pd.read_csv('accident.csv')

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

plt.show()
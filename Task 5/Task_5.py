'''
- GEOGRAPHICAL CLUSTERING
For task 5, we will focus on these things:
    + Identify the top 10 accident-prone locations (NODE_ID)
    + For these locations, analyze the most common ACCIDENT_TYPE_DESC and DCA_DESC
      --> save as Task_5_accident_trends_locations.png and Task_5_individual_location_analysis.png
    + Create visualizations comparing these high-risk locations in terms of severity and time patterns
      --> save as Task_5_hotspot_comparison.png
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Read the dataset into a dataframe
df = pd.read_csv('../dataset/accident.csv')

'''
    REQUIREMENT 1: - Identify the TOP 10 accident-prone locations (NODE_ID)
                   - And then for these 10 locations, analyze the most common ACCIDENT_TYPE_DESC and DCA_DESC
                     --> save as Task_5_hotspots.png
'''
# Firstly drop the row with NaN values at these targeted columns
cleaned_df = df.dropna(subset=['NODE_ID', 'ACCIDENT_TYPE_DESC', 'DCA_DESC'])

# Get top 10 NODE_IDs with most accidents
top_10_node = (
    cleaned_df['NODE_ID']
    .value_counts()
    .head(10)
    .index
    .tolist()
)

# Filter the df to just get the row with the NODE_ID in top 10 NODE_ID
filtered_df = df.loc[df['NODE_ID'].isin(top_10_node)]

# Drop rows with NaN in required columns
cleaned_df = df.dropna(subset=['NODE_ID', 'ACCIDENT_TYPE_DESC', 'DCA_DESC'])

# Get top 10 accident-prone NODE_IDs
top_10_node = (
    cleaned_df['NODE_ID']
    .value_counts()
    .head(10)
    .index
    .tolist()
)

# Filter the dataframe to include only rows from top 10 nodes
filtered_df = cleaned_df[cleaned_df['NODE_ID'].isin(top_10_node)]


# First, let's create a function to categorize DCA_DESC
def categorize_dca(desc):
    # Intersection collisions
    intersection = ['RIGHT THROUGH', 'CROSS TRAFFIC(INTERSECTIONS ONLY)', 'RIGHT NEAR (INTERSECTIONS ONLY)',
                    'RIGHT FAR (INTERSECTIONS ONLY)', 'LEFT NEAR (INTERSECTIONS ONLY)', 'LEFT FAR (INTERSECTIONS ONLY)',
                    'OTHER ADJACENT (INTERSECTIONS ONLY)', 'RIGHT/LEFT FAR (INTERSECTIONS ONLY)']

    # Pedestrian-related accidents
    pedestrian = ['FAR SIDE. PED HIT BY VEHICLE FROM THE LEFT', 'PED NEAR SIDE. PED HIT BY VEHICLE FROM THE RIGHT.',
                  'VEH STRIKES PED ON FOOTPATH/MEDIAN/TRAFFIC ISLAND.',
                  'PED ON FOOTHPATH STRUCK BY VEHENTERING/LEAVING DRIVEWAY.',
                  'PED PLAYING/LYING/WORKING/STANDING ON CARRIAGEWAY.',
                  'ANY MANOEUVRE INVOLVING PED NOT INCLUDED IN DCAs 100-108.']

    # Lane change/sideswipe accidents
    lane_change = ['RIGHT TURN SIDESWIPE', 'LEFT TURN SIDESWIPE', 'LANE SIDE SWIPE (VEHICLES IN PARALLEL LANES)',
                   'LANE CHANGE LEFT (NOT OVERTAKING)', 'LANE CHANGE RIGHT (NOT OVERTAKING)',
                   'CUTTING IN (OVERTAKING)', 'U TURN']

    # Off-road/loss of control accidents
    off_road = ['RIGHT OFF CARRIAGEWAY INTO OBJECT/PARKED VEHICLE', 'LEFT OFF CARRIAGEWAY INTO OBJECT/PARKED VEHICLE',
                'OFF CARRIAGEWAY TO LEFT', 'OFF CARRIAGEWAY TO RIGHT', 'OFF RIGHT BEND INTO OBJECT/PARKED VEHICLE',
                'OFF END OF ROAD/T-INTERSECTION.', 'OUT OF CONTROL ON CARRIAGEWAY (ON STRAIGHT)',
                'OUT OF CONTROL ON CARRIAGEWAY (ON BEND)', 'OTHER ACCIDENTS-OFF STRAIGHT NOT INCLUDED IN DCAs 170-175',
                'OTHER ACCIDENTS ON CURVE NOT INCLUDED IN DCAs 180-184']

    # Rear-end collisions
    rear_end = ['REAR END(VEHICLES IN SAME LANE)', 'LEFT REAR', 'RIGHT REAR.', 'PULLING OUT -REAR END']

    # Parking/stationary vehicle related
    parking = ['VEHICLE STRIKES DOOR OF PARKED/STATIONARY VEHICLE', 'LEAVING PARKING', 'PARKED CAR RUN AWAY']

    # Head-on collisions
    head_on = ['HEAD ON (NOT OVERTAKING)']

    # Other vehicle maneuvers
    other_vehicle = ['VEHICLE OFF FOOTPATH STRIKES VEH ON CARRIAGEWAY',
                     'VEHICLE STRIKES ANOTHER VEH WHILE EMERGING FROM DRIVEWAY',
                     'OTHER MANOEUVRING NOT INCLUDED IN DCAs 140-148',
                     'OTHER OPPOSING MANOEUVRES NOT INCLUDED IN DCAs 120-125.']

    # Other/Miscellaneous
    misc = ['FELL IN/FROM VEHICLE', 'STRUCK TRAIN', 'OTHER ACCIDENTS NOT CLASSIFIABLE ELSEWHERE']

    if desc in intersection:
        return 'Intersection Collision'
    elif desc in pedestrian:
        return 'Pedestrian Accident'
    elif desc in lane_change:
        return 'Lane Change/Sideswipe'
    elif desc in off_road:
        return 'Off-road/Loss of Control'
    elif desc in rear_end:
        return 'Rear-end Collision'
    elif desc in parking:
        return 'Parking/Stationary Vehicle'
    elif desc in head_on:
        return 'Head-on Collision'
    elif desc in other_vehicle:
        return 'Other Vehicle Maneuver'
    elif desc in misc:
        return 'Miscellaneous'
    else:
        return 'Other'


# Function to categorize ACCIDENT_TYPE_DESC
def categorize_accident_type(desc):
    vehicle_collision = ['Collision with vehicle']
    pedestrian = ['Struck Pedestrian']
    fixed_object = ['Collision with a fixed object', 'collision with some other object']
    no_collision = ['No collision and no object struck', 'Vehicle overturned (no collision)']
    other = ['Fall from or in moving vehicle', 'Other accident']

    if desc in vehicle_collision:
        return 'Vehicle Collision'
    elif desc in pedestrian:
        return 'Pedestrian Incident'
    elif desc in fixed_object:
        return 'Fixed Object Collision'
    elif desc in no_collision:
        return 'No Collision'
    elif desc in other:
        return 'Other'
    else:
        return 'Unclassified'

# Apply categorization to our filtered dataframe
filtered_df['DCA_Category'] = filtered_df['DCA_DESC'].apply(categorize_dca)
filtered_df['ACCIDENT_TYPE_Category'] = filtered_df['ACCIDENT_TYPE_DESC'].apply(categorize_accident_type)

# Set the style
plt.style.use('ggplot')
sns.set_palette("colorblind")

# FIGURE 1: Trends across all 10 locations
plt.figure(figsize=(16, 10))

# Create a subplot for DCA categories
plt.subplot(2, 1, 1)
dca_counts = filtered_df.groupby(['NODE_ID', 'DCA_Category']).size().unstack().fillna(0)
# Sort the columns by total frequency
dca_columns_order = filtered_df['DCA_Category'].value_counts().index.tolist()
dca_counts = dca_counts.reindex(columns=dca_columns_order)

# Plot stacked bar for DCA categories
dca_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Distribution of DCA Categories Across Top 10 Accident-Prone Locations', fontsize=14)
plt.xlabel('NODE_ID')
plt.ylabel('Number of Accidents')
plt.legend(title='DCA Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

# Create a subplot for Accident Type categories
plt.subplot(2, 1, 2)
accident_type_counts = filtered_df.groupby(['NODE_ID', 'ACCIDENT_TYPE_Category']).size().unstack().fillna(0)
# Sort the columns by total frequency
acc_type_columns_order = filtered_df['ACCIDENT_TYPE_Category'].value_counts().index.tolist()
accident_type_counts = accident_type_counts.reindex(columns=acc_type_columns_order)

# Plot stacked bar for Accident Type categories
accident_type_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Distribution of Accident Type Categories Across Top 10 Accident-Prone Locations', fontsize=14)
plt.xlabel('NODE_ID')
plt.ylabel('Number of Accidents')
plt.legend(title='Accident Type Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('Task_5_accident_trends_across_locations.png', dpi=300, bbox_inches='tight')

# FIGURE 2: Individual location analysis
# Create a figure with subplots for each location
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(4, 3, figure=fig)

# Calculate total accidents per node for reference
node_accident_counts = filtered_df['NODE_ID'].value_counts()

# Create subplots for each NODE_ID
for i, node_id in enumerate(top_10_node):
    # Get data for this location
    node_data = filtered_df[filtered_df['NODE_ID'] == node_id]

    # Calculate percentages for DCA categories
    dca_percentages = node_data['DCA_Category'].value_counts(normalize=True) * 100
    # Get top 3 DCA categories
    top_dca = dca_percentages.head(3)

    # Calculate percentages for Accident Type categories
    acc_type_percentages = node_data['ACCIDENT_TYPE_Category'].value_counts(normalize=True) * 100
    # Get top 3 Accident Type categories
    top_acc_type = acc_type_percentages.head(3)

    # Position in the grid
    row = i // 3
    col = i % 3

    # Create subplot
    ax = fig.add_subplot(gs[row, col])

    # Create a horizontal bar chart for DCA and Accident Type
    categories = []
    percentages = []
    colors = []

    # Add DCA categories
    for cat, pct in top_dca.items():
        categories.append(f"DCA: {cat}")
        percentages.append(pct)
        colors.append('skyblue')

    # Add Accident Type categories
    for cat, pct in top_acc_type.items():
        categories.append(f"Type: {cat}")
        percentages.append(pct)
        colors.append('lightcoral')

    # Create horizontal bar chart
    y_pos = np.arange(len(categories))
    ax.barh(y_pos, percentages, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)

    # Add percentage labels
    for i, v in enumerate(percentages):
        ax.text(v + 1, i, f"{v:.1f}%", va='center')

    # Set title and labels
    total_accidents = node_accident_counts[node_id]
    ax.set_title(f"NODE_ID: {node_id} (Total: {total_accidents} accidents)", fontsize=12)
    ax.set_xlabel('Percentage (%)')
    ax.set_xlim([0, 100])

plt.tight_layout()
plt.savefig('Task_5_individual_location_analysis.png', dpi=300, bbox_inches='tight')

'''
    REQUIREMENT 2: Create visualizations comparing these high-risk locations in terms of severity and time patterns
                   --> save as Task_5_hotspot_comparison.png
'''
# This function is to process time patterns ACCIDENT_TIME
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


'''
    REQUIREMENT 2: Create visualizations comparing these high-risk locations in terms of severity and time patterns
                   --> save as Task_5_hotspot_comparison.png
'''
# Apply the time of day function to categorize accident times
filtered_df.loc[:, 'TIME_OF_DAY'] = filtered_df['ACCIDENT_TIME'].apply(get_time_of_day)

# Set the style
plt.style.use('ggplot')
sns.set_palette("colorblind")

# Create figure with 2 rows
fig, axes = plt.subplots(2, 1, figsize=(16, 14))

# Panel 1: Severity Analysis
# Create a crosstab of NODE_ID vs Severity
severity_crosstab = pd.crosstab(filtered_df['NODE_ID'], filtered_df['SEVERITY'])

# Calculate the percentage of each severity level for each NODE_ID
severity_percentage = severity_crosstab.div(severity_crosstab.sum(axis=1), axis=0) * 100

# Sort NODE_IDs by total accident count for consistency
node_order = filtered_df['NODE_ID'].value_counts().index.tolist()
severity_percentage = severity_percentage.reindex(node_order)

# Create a stacked bar chart for severity
severity_percentage.plot(kind='bar', stacked=True, ax=axes[0], colormap='RdYlGn_r')
axes[0].set_title('Distribution of Accident Severity Across Top 10 High-Risk Locations', fontsize=16)
axes[0].set_xlabel('NODE_ID', fontsize=12)
axes[0].set_ylabel('Percentage (%)', fontsize=12)
axes[0].legend(title='Severity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

# Add total accident count as text above each bar
for i, node in enumerate(node_order):
    total = filtered_df[filtered_df['NODE_ID'] == node].shape[0]
    axes[0].text(i, 105, f'n={total}', ha='center', fontsize=10)

# Panel 2: Time Pattern Analysis
# Create a crosstab of NODE_ID vs Time of Day
time_crosstab = pd.crosstab(filtered_df['NODE_ID'], filtered_df['TIME_OF_DAY'])

# Calculate the percentage of each time category for each NODE_ID
time_percentage = time_crosstab.div(time_crosstab.sum(axis=1), axis=0) * 100

# Reorder columns in a logical sequence
time_order = ['Morning', 'Afternoon', 'Evening', 'Late Night', 'Unknown']
time_percentage = time_percentage.reindex(columns=[col for col in time_order if col in time_percentage.columns])

# Reorder rows by total accident count
time_percentage = time_percentage.reindex(node_order)

# Create a stacked bar chart for time of day
time_percentage.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis')
axes[1].set_title('Distribution of Accident Time of Day Across Top 10 High-Risk Locations', fontsize=16)
axes[1].set_xlabel('NODE_ID', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].legend(title='Time of Day', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

# Add hourly heatmap below the main plots
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Make room for the heatmap

# Create a third subplot for hourly distribution heatmap
ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.15])  # [left, bottom, width, height]


# Extract hour from ACCIDENT_TIME and create hourly bins
def extract_hour(time_str):
    try:
        return int(time_str[:2])
    except (ValueError, TypeError):
        return np.nan


filtered_df.loc[:, 'HOUR'] = filtered_df['ACCIDENT_TIME'].apply(extract_hour)

# Create a pivot table for the heatmap: NODE_ID vs Hour
hourly_data = pd.pivot_table(
    filtered_df.dropna(subset=['HOUR']),
    values='SEVERITY',  # Using SEVERITY just to count occurrences
    index='NODE_ID',
    columns='HOUR',
    aggfunc='count',
    fill_value=0
)

# Reorder rows by total accident count
hourly_data = hourly_data.reindex(node_order)

# Create the heatmap
sns.heatmap(hourly_data, cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Number of Accidents'})
ax3.set_title('Hourly Distribution of Accidents by Location', fontsize=14)
ax3.set_xlabel('Hour of Day (24-hour format)', fontsize=12)
ax3.set_ylabel('NODE_ID', fontsize=12)

# Add a summary table showing key insights
# Get the most dangerous time period and most common severity for each location
summary_data = []
for node in node_order:
    node_data = filtered_df[filtered_df['NODE_ID'] == node]

    # Most common time of day - safely getting the values using .iloc
    time_counts = node_data['TIME_OF_DAY'].value_counts(normalize=True)
    if not time_counts.empty:
        common_time = time_counts.index[0]
        time_pct = time_counts.iloc[0] * 100
    else:
        common_time = "Unknown"
        time_pct = 0

    # Most common severity - safely getting the values using .iloc
    severity_counts = node_data['SEVERITY'].value_counts(normalize=True)
    if not severity_counts.empty:
        common_severity = severity_counts.index[0]
        severity_pct = severity_counts.iloc[0] * 100
    else:
        common_severity = "Unknown"
        severity_pct = 0

    # Most dangerous hour - safely getting the values using .iloc
    hour_counts = node_data['HOUR'].value_counts(normalize=True)
    if not hour_counts.empty:
        dangerous_hour = hour_counts.index[0]
        hour_pct = hour_counts.iloc[0] * 100
    else:
        dangerous_hour = "Unknown"
        hour_pct = 0

    summary_data.append({
        'NODE_ID': node,
        'Total_Accidents': node_data.shape[0],
        'Common_Time': common_time,
        'Time_%': time_pct,
        'Dangerous_Hour': dangerous_hour,
        'Hour_%': hour_pct,
        'Common_Severity': common_severity,
        'Severity_%': severity_pct
    })

# Save the summary to a CSV file
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('Task_5_hotspot_comparison.csv')

# Save the figure
plt.savefig('Task_5_hotspot_comparison.png', dpi=300, bbox_inches='tight')
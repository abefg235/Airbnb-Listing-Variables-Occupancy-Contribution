############################################
############################################
# CALC_OCCUPANCY_RATE.PY
# Input: 
#    - listings_df_name: filepath and name of listings CSV, downloaded from 
#                        Inside Airbnb or the output of augment_df.py
#    - review_df_name: filepath and name of reviews CSV, downloaded from 
#                      Inside Airbnb
# Output:
#    - out.csv: a CSV containing listings data and occupancy rate
#
# The following code will calculate each listing's occupancy rate using Inside
# Airbnb's San Francisco Model (http://insideairbnb.com/about.html#disclaimers).
# Occupancy rate will be added as a new variable to listings dataframe.
############################################
############################################

import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# filepath and name of listings CSV, either downloaded from Inside Airbnb
# or the output of augment_df.py
listings_df_name = sys.argv[1]

# filepath and name of reviews CSV, downloaded from Inside Airbnb
review_df_name = sys.argv[2]

############################################
# INITIAL PARAMETERS #######################
############################################
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
avg_length_of_stay = 3 # nights
review_rate = 0.5

############################################
# HELPER FUNCTIONS #########################
############################################
def calc_occupancy_rate(data):
    num_of_reviews = data['number_of_reviews']
    dates_listed = data['listing_time']
    minimum_nights = data['minimum_nights']

    if dates_listed == 0:
        return np.nan
    elif num_of_reviews == 0:
        return 0
    num_of_occupants = num_of_reviews/review_rate
    days_occupied = num_of_occupants*max(avg_length_of_stay,minimum_nights)
    occupancy_rate = days_occupied/dates_listed
    if occupancy_rate > 0.7:
        return 0.7
    return occupancy_rate

def calc_listing_time(data):
    listing_id = int(data['id'])
    reviews = reviews_df.loc[reviews_df['listing_id'] == listing_id]

    if reviews.empty: return 0

    date0 = str(reviews['date'].max())
    date1 = str(reviews['date'].min())

    dates = date0.split('-')
    d0 = date(int(dates[0]), int(dates[1]), int(dates[2]))
    dates = date1.split('-')
    d1 = date(int(dates[0]), int(dates[1]), int(dates[2]))

    delta = d0 - d1
    return delta.days

############################################
# CALCULATE OCCUPANCY RATE #################
############################################
num_listings_df = pd.read_csv(listings_df_name)
reviews_df = pd.read_csv(review_df_name)

print(num_listings_df.shape)
num_listings_df = num_listings_df.select_dtypes(include=numerics)
print(num_listings_df.columns)

# Do NOT apply dropna because some of the columns we need have empty values

# Calculate occupancy rate
num_listings_df['listing_time'] = num_listings_df.apply(calc_listing_time, axis=1) 
num_listings_df['occupancy_rate'] = num_listings_df.apply(calc_occupancy_rate, axis=1) 

# Sanity check
print('Orig shape: '+str(num_listings_df.shape))

# Now we can remove all the rows with NaN 
num_listings_df.dropna(axis=0, inplace=True)
print('After shape: '+str(num_listings_df.shape))

# Dump DF for testing and in preparation of next steps
num_listings_df.to_csv('out.csv', index=False)

# Print something to confirm code has finished running
print("\nEND\n")

############################################
############################################
# AUGMENT_DF.PY
# Input: 
#    - listings_df_name: filepath and name of listings CSV, 
#                         downloaded from Inside Airbnb
# Output:
#    - out.csv: a CSV containing cleaned and filtered listings data
#
# The following code will remove variables found to be unusable (either due
# to the nature of the data or due to data scarcity) and transform certain
# text variables to be numeric. This processing is done to prepare data 
# for linear regression to identify variables most highly correlated with
# listings' popularity (as measured by occupancy rate)
############################################
############################################


import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

# filepath and name of listings CSV, downloaded from Inside Airbnb
listings_df_name = sys.argv[1]

############################################
# INITIAL PARAMETERS #######################
############################################
# identified manually, unfortunately

prices = ["security_deposit","cleaning_fee","extra_people"]
col_text = ["name","summary","space","description","neighborhood_overview","notes",
            "transit","access", "interaction", "house_rules","host_name","host_location",
            "host_about"]
col_location = ["street","neighbourhood","neighbourhood_cleansed",
                "neighbourhood_group_cleansed","city","state","zipcode","market","smart_location",
                "country_code","country","jurisdiction_names"]
col_no_use = ["listing_url","scrape_id","last_scraped","picture_url","xl_picture_url",
              "host_id","host_url","host_thumbnail_url","host_picture_url"]
col_optional_use = ["weekly_price","monthly_price"]
col_too_little_data = ["experiences_offered","host_acceptance_rate","thumbnail_url","medium_url",
                       "host_neighbourhood","square_feet","license"]
col_assume_0_unless_otherwise = ["accommodates","bathrooms","bedrooms","beds","security_deposit",
                                 "cleaning_fee"]

############################################
# HELPER FUNCTIONS #########################
############################################
# We will be running a regression, which takes numerical inputs. Therefore,
# must map text to numerical values.

def translate_response_time(val):
    if val == 'within an hour':
        return 1
    elif val == 'within a few hours':
        return 3
    elif val == 'within a day':
        return 24
    elif val == 'a few days or more':
        return 72
    else:
        return np.nan

def is_binary(df):
    elements = list(df.unique())
    if len(elements) == 2:
        if 't' in elements and 'f' in elements:
            return True
    elif len(elements) == 3:
        if 't' in elements and 'f' in elements and np.nan in elements:
            return True
    return False

def translate_tf_to_10(val):
    # keep nulls as NaN
    if val == 't':
        return 1
    elif val == 'f':
        return 0

def translate_price_to_float(val):
    # keep nulls as NaN
    if str(val) == "nan":
        return np.nan
    elif "$" in val:
        val = val[1:]
    return float(val.replace(',', ''))

def translate_percent_to_decimal(val):
    # keep nulls as NaN
    if str(val) == "nan":
        return np.nan
    elif "%" in val:
        val = val[:-1]
    return float(val)/100

def translate_cancellation_policy(val):
    # Policies are ranked by strictness
    # keep nulls as NaN
    policy_strictness = 0
    if str(val) == 'flexible':
        policy_strictness = 1
    elif str(val) == 'moderate':
        policy_strictness = 2
    elif str(val) == 'strict_14_with_grace_period':
        policy_strictness = 3
    elif str(val) == 'strict':
        policy_strictness = 4
    elif str(val) == 'super_strict_30':
        policy_strictness = 5
    elif str(val) == 'super_strict_60':
        policy_strictness = 6
    elif str(val) == 'long_term':
        policy_strictness = 7
    elif str(val) == "nan":
        policy_strictness = np.nan
    else:
        print("Cancellation policy: "+str(val))
        print("AN UNEXPECTED CANCELLATION POLICY STRICTNESS!\n")
    return policy_strictness

############################################
# ANALYSIS BEGINS ##########################
############################################
num_listings_df = pd.read_csv(listings_df_name)

# Drop variables that have been identified to be not useful
num_listings_df.drop(columns=col_text,inplace=True)
num_listings_df.drop(columns=col_location,inplace=True)
num_listings_df.drop(columns=col_no_use,inplace=True)
num_listings_df.drop(columns=col_too_little_data,inplace=True)
num_listings_df.drop(columns=col_optional_use, inplace=True)

# Sanity check - confirm that correct columns were dropped
print(num_listings_df.columns)

# Translate potentially useful text variables into numerical variables
num_listings_df["num_host_verifications"] = num_listings_df["host_verifications"].apply(lambda x: 
                                            len(x.split(',')) if x!= '[]' else 0)
num_listings_df["num_amenities"] = num_listings_df["amenities"].apply(lambda x:
                                   len(x.split(',')) if x != '{}' else 0)
num_listings_df["host_response_rate"] = num_listings_df["host_response_rate"].apply(translate_percent_to_decimal)
num_listings_df['cancellation_strictness'] = num_listings_df["cancellation_policy"].apply(translate_cancellation_policy)
num_listings_df['host_response_time'] = num_listings_df["host_response_time"].apply(translate_response_time)

# One hot encoding for categorical variables
temp_df = pd.get_dummies(num_listings_df['room_type'],prefix='room_type',dummy_na=True)
num_listings_df = pd.concat([num_listings_df,temp_df], axis=1)
temp_df = pd.get_dummies(num_listings_df['bed_type'],prefix='bed_type',dummy_na=True)
num_listings_df = pd.concat([num_listings_df,temp_df], axis=1)

# Sanity check
print(num_listings_df.shape)

# Some variables take only one value other than null (NaN). These variables
# are not useful for regression. Identify and remove them from dataset
cols_no_useful_info = []
for col in num_listings_df.columns:
    elements = num_listings_df[col].unique()

    if len(elements) == 1:
        cols_no_useful_info.append(col)
        continue
    elif np.nan in list(elements) and len(elements) == 2:
        cols_no_useful_info.append(col)
        continue

    if is_binary(num_listings_df[col]):
        num_listings_df[col] = num_listings_df[col].apply(translate_tf_to_10)
        continue
    if "price" in col or col in prices:
        num_listings_df[col] = num_listings_df[col].apply(translate_price_to_float)
print("Columns with no useful info:\n")
print(cols_no_useful_info)
num_listings_df.drop(columns=cols_no_useful_info, inplace=True)

# For variables assumed to be zero if null, fill in the zeroes!
num_listings_df[col_assume_0_unless_otherwise] = num_listings_df[col_assume_0_unless_otherwise].fillna(value=0)

# sanity check, to confirm that unwanted variables have been dropped
print(num_listings_df.shape)

# Dump DF for testing and in preparation of next steps
num_listings_df.to_csv('out.csv', index=False)

# Print something to confirm code has finished running
print('\nEND\n')


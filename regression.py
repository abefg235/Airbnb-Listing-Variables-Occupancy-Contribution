############################################
############################################
# REGRESSION.PY
# Input: 
#    - data_df_name: output of calc_occupancy_rate.py
# Output: N/A. Results of regressions will be printed
#
# The following code will run a linear regression between various Airbnb 
# variables and the occupancy rate (used to measure 'popularity'), calculate
# statistics to quantify quanlity of fit (adjusted R^2, PRESS and Mallow's Cp).
# Afterwards, calculate and filter variables by variance inflation factor (VIF) 
# to reduce collinearity between variables. Re-run regression and recalculate 
# statistics.
############################################
############################################

import math
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model

#####################################
# PARAMETERS PASSED THROUGH #########
#####################################
# output of calc_occupancy_rate.py
data_df_name = sys.argv[1]

# latitude and longitude information - will be used to 
# filter for listings from specific neighbourhoods
lat0 = float(sys.argv[2])
lat1 = float(sys.argv[3])
long0 = float(sys.argv[4])
long1 = float(sys.argv[5])

#####################################
# PARAMETERS ########################
#####################################
# Two sets of columns in listings DF that are not used for linear regression
X_cols_to_delete = ["id","scrape_id","host_id","host_listings_count",
                    "host_total_listings_count","latitude","longitude",
                    "occupancy_rate","listing_time","number_of_reviews",
                    "number_of_reviews_ltm","reviews_per_month"]
cols_dont_use = ["weekly_price","monthly_price","security_deposit","cleaning_fee"]

# Cutoff for VIF. Any variables with VIFs > VIF_too_high will be omitted from
# second regression
VIF_too_high = 4

keep_inf_VIFs = True

# Significance value, used for hypothesis testing
alpha = 0.05


#####################################
# HELPER FUNCTIONS ##################
#####################################
def calc_y_hat(X0,y,params):
    # sklearn will sometimes assume one of the variables is a constant value.
    # If there is no constant value, then it will add one. If one is added,
    # then number and order of parameters will change, causing y_hat to be
    # calculated incorrectly, hence reason for the following if-statements
    
    y_hat = np.zeros(y.shape[0])
    
    if len(X0.columns) == len(params):
        # there was no constant
        i = 0
    elif len(X0.columns)+1 == len(params):
        # there is a constant
        i = 1
        y_hat += params[0]
    else:
        print("WARNING! THERE ARE SOME EXTRA COEFFICIENTS!\n")

    for column in X0.columns:
        if column == 'const': continue
        array = np.array(X0[column])
        coef = params[i]
        y_hat += np.multiply(coef,array)
        i+=1
    return y_hat

def calculate_stats(X_new,y,residuals):
    hat_matrix = np.dot(X_new,np.dot(np.linalg.inv(np.dot(np.transpose(X_new),X_new)),np.transpose(X_new)))
    
    h_ii = np.diagonal(hat_matrix)
    e__i = np.divide(residuals,np.subtract(1,h_ii))
    PRESS = np.sum(np.square(e__i))
    
    ESS = np.sum(np.square(residuals))
    y_avg = np.average(y)
    TSS = np.sum(np.square(y-y_avg))
    R2 = 1-ESS/TSS
    print("ESS = "+str(ESS))
    print("RSS = "+str(TSS-ESS))
    print("TSS = "+str(TSS))
    print("R^2 = "+str(R2))
    print("PRESS = "+str(PRESS))
    
    n = len(y)
    P2 = 1-(PRESS/TSS)*((n-1)/n)**2
    print("P^2 = "+str(P2))


#####################################
# ANALYSIS STARTS HERE ##############
#####################################
data_df = pd.read_csv(data_df_name)

# filter based on latitude and longitude to obtain data for a particular
# neighbourhood in each city
data_df['latitude'] = data_df['latitude'].astype(float)
data_df['longitude'] = data_df['longitude'].astype(float)
data_df = data_df.loc[(data_df['latitude']>lat0) | (data_df['latitude']==lat0)]
data_df = data_df.loc[(data_df['latitude']<lat1) | (data_df['latitude']==lat1)]
data_df = data_df.loc[(data_df['longitude']>long0) | (data_df['longitude']==long0)]
data_df = data_df.loc[(data_df['longitude']<long1) | (data_df['longitude']==long1)]
data_df.drop(columns=['latitude','longitude'])

# Sanity check
print('Survived location filtering!')
print("new shape of data_df: "+str(data_df.shape))

# TODO: this function also appears in augment_df.py. Should replace this code
# with a helper function also used by augment_df.py
cols_no_useful_info = []
for col in data_df.columns:
    elements = data_df[col].unique()

    if len(elements) == 1:
        cols_no_useful_info.append(col)
        continue
    elif np.nan in list(elements) and len(elements) == 2:
        cols_no_useful_info.append(col)
        continue
data_df.drop(columns=cols_no_useful_info, inplace=True)

# Sanity check
print("new shape of data_df: "+str(data_df.shape))

# y is the dependent variable in this regression
y = data_df['occupancy_rate']
print("max occ rate: "+str(y.max()))
print("min occ rate: "+str(y.min()))

# X contains the independent variables 
X = data_df.drop(columns=X_cols_to_delete, errors='ignore')
X = X.drop(columns=cols_dont_use, errors='ignore')
X.dropna(inplace=True)

# normalize
X = (X-X.min())/(X.max()-X.min())
y = (y-y.min())/(y.max()-y.min())
X.to_csv("X_norm.csv") # for error checking

###################################
# Run original model ##############
###################################
X_new = sm.add_constant(X)
print(X_new.columns)

# run initial regreesion model and print results
model_0 = sm.OLS(y, X_new)
results_0 = model_0.fit()
print(results_0.summary())

params_0 = results_0.params
print(params_0)

y_0_hat = calc_y_hat(X,y,params_0)
residuals_0 = y-y_0_hat

print("\nStatistics for initial regression")
# print statistics to help quantify quality of fit
calculate_stats(X_new,y,residuals_0)
n = len(y)
p_0 = len(params_0)
ESS_0 = np.sum(np.square(residuals_0))
sk2 = ESS_0/(n-p_0)
print('sk^2 = '+str(sk2))

# print statistically significant relationships (i.e. variables 
# with p-value < alpha)
pvalue = results_0.pvalues.le(alpha).to_frame()
pvalue_bad = list(pvalue.loc[pvalue[0] == False].index)
X_temp = X_new.drop(columns=pvalue_bad,errors='ignore')
print(X_temp.columns)

###################################
# VIFs ############################
###################################
# now, remove collinear variables using VIF
VIFs = pd.Series([variance_inflation_factor(X_new.values, i)
                  for i in range(X_new.shape[1])],
                  index=X_new.columns)
print("VIFs:\n"+str(VIFs))

# remove all variables with VIF > cut off (VIF_too_high)
VIFs_ok_df = VIFs.le(VIF_too_high).to_frame()
print(VIFs_ok_df)
VIFs_bad = list(VIFs_ok_df.loc[VIFs_ok_df[0] == False].index)

if keep_inf_VIFs:
    VIFs_inf_df = VIFs.ge(99999999).to_frame()
    VIFs_inf = list(VIFs_inf_df.loc[VIFs_inf_df[0] ==True].index)
    VIFs_bad = list(set(VIFs_bad) - set(VIFs_inf))

print("Truly bad VIFs")
print(VIFs_bad)

#####################################
# Run regression without collinear variables
#####################################
# remove collinear variables
X0 = X_new.drop(columns=VIFs_bad)
X0 = X0.drop(columns=['const'], errors='ignore')
X0_new = sm.add_constant(X0)

# Run second regression model, this time without collinear variables
model_1 = sm.OLS(y, X0_new)
results_1 = model_1.fit()
print(results_1.summary())

# calculate the Mallow's Cp to quantify quality of fit
params = results_1.params
y_hat = calc_y_hat(X0,y,params)

residuals = y-y_hat

print("\ncalculating statistics after VIF filter")
calculate_stats(X0_new, y, residuals)

p = len(params)
Cp = np.sum(np.square(residuals))/sk2-(n-2*p)
print("Cp = "+str(Cp))

# print statistically significant relationships (i.e. variables 
# with p-value < alpha)
pvalue = results_1.pvalues.le(alpha).to_frame()
pvalue_bad = list(pvalue.loc[pvalue[0] == False].index)
X1 = X0.drop(columns=pvalue_bad,errors='ignore')
print(X1.columns)

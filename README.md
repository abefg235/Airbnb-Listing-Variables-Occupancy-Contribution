# What Variables Contribute Most to Airbnb Listings’ Occupancy Rate?
A portion of a course project from 2019. Other parts are not included here because I didn't work on them.

The objective of this project was to analyze Airbnb property listings to determine which factors most strongly influence occupancy rate.

The input data are listing.csv and reviews.csv obtained from Inside Airbnb. Data is cleaned, occupancy rate is calculated, and a linear regression model is ran. Looking at the coefficients, standard deviations, and p-values tells us which variables have a strong, statistically significant relationship with occupancy rate. Various statistics (ex. ajusted R^2) quantifies the quality of fit.

## My results
I actually ran this code was ran on data for New York, NY (NYC) and Jersey City, NJ. Note that these cities were chosen because they were geographically close, making them interesting to compare. 
I found that the variables that affects occupancy rate the most were:
- host_response_time (how quickly does the host respond?)
- accommodates (how many people can the listing accomodate?)
- price ($$$)
- review_scores_communication (how well did the host communicate?)
- instant_bookable (do guests need to wait for host to confirm their booking?)

Interestingly, NYC listings with 'Real beds' corresponded to higher occupancy rates, while in Jersey City, 'Futons' did. Also, better reviews for communication with host corresponded with higher occupancy rates in Jersey City and with lower occupanyc rates in NYC.

## The code
Order of execution:
1) augment_df.py
2) calc_occupancy_rate.py
3) regression.py

### augment_df
Cleans data

To run: *python augment_df.py [fpath to listings CSV]*

### calc_occupancy_rate
Calculates the occupancy rate, as per Inside Airbnb's San Francisco Model (http://insideairbnb.com/about.html#disclaimers)

To run: *python calc_occupancy_rate.py [fpath to output from augment_df.py] [fpath to reviews CSV]*
  
### regression.py
Runs linear regression model on the filtered data, with occupancy rate as the dependent variable. Afterwards, calculates various statistics (ex. adjusted R^2, PRESS and Mallow's Cp) to quantify the quality of the model. Filters variables by variance inflation factor (VIF) to reduce collinearity, reruns linear regression model, and recalculates statistics.

To run: *python regression.py [fpath to output from calc_occupancy_rate.py] [lat0] [lat1] [long0] [long1]*

(lat0, lat1) is the range of latitude that a listing must fall in for the data to be analysed. (long0, long1) is the range of longitude. The group added these filters so we have the option to focus our analysis on certain neighbourhoods (roughly).

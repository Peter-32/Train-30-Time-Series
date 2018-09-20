# Spot-Check-RF-Vs-AutoARIMA

_

### Prerequisites

_

### Installing

_

### Goal

Run a 10 small random forest times on each of 30 univariate time series to predict the next 10 days.  Each model predicts one of the days ahead.  Compare the RMSE to AutoARIMA.  View the plots of each

### Data Sources

- https://www.kaggle.com/felixzhao/productdemandforecasting/home
- historical_product_demand.csv

### Data Descriptions

- Descriptions from: https://www.kaggle.com/felixzhao/productdemandforecasting/home
- Product_CodeThe product name encoded
- WarehouseWarehouse name encoded
- Product_CategoryProduct Category for each Product_Code encoded
- DateThe date customer needs the product
- Order_Demandsingle order qty


### Steps

- ETL
  - X Use a text editor
  - X Load the data into pandas
  - X Format the datetime column
  - X Create the model key
  - X Choose 10 models with the most records
  - X Save to a table
  - Comment out code
- Exploration
  - Load from table
  - X Excel
  - X Date as index
  - X Create a function to view the 10 plots
  - X Plot all 10
  - X Check for preferred min/max dates.  Same or different for each time series?
  - X Consider capping at 95% percentiles
  - X Consider logging the data
  - X thoughts on imputing values?
  - X Check for missing values / infinity
  - X Comment out code  
- Prepare Data
  - X Load from table
  - X Loop over model keys (break after one iteration for dev)  
  - X Load model key and drop that column
  - X Set max and min date
  - X Fill missing dates and date as index
  - X Optionally cap the data at 95% percentiles
  - X Fix missing values or infinity
  - X View the plot and save it
  - X Optionally log the data, if so, view the plot and save it
  - X Consider if it looks stationary (consider seasonality & trend); regardless don't try to fix it for now
  - X Add the model_key back to the df
  - X Combine all data
  - X Plot the 10 plots and save the image
  - X Save to table
- Spot Check Algorithms
  - X Load from table
  - X Loop over model keys (break after one iteration for dev)  
  - X Loop over days forecast ahead
  - X Load model key and drop that column
  - X Date as index
  - X Get the y values
  - X Use last 10 days as the test set
  - X Chose not to use CV because I don't want to tune the algorithms
  - X Check the average and persistence MAPE 1 day ahead
  - Check Auto ARIMA MAPE 1 day ahead
  - Check lag 8 Random Forest MAPE 1 day ahead
  - Record all results in a pandas then CSV file
- Improve
  - Skip
- Post
  - Jupyter Notebook HTML to document the steps (basic, isolation, understandable, & plots)
  - Basic:
    - Only focus on the necessary things
  - Isolation:
    - Don't reference other projects
  - Understandable
    - Make no assumptions about what others know
  - Plots:
    - the prediction for all 10 plots 1.png, ... 10.png.  One line for actual, AUTOArima, and RF
    - Title the plots as the RMSE for each line, and a line for the best RMSE.  Add a legend.  Include a baseline RMSE in this title.
- End of project
# ts-train-30-models

_

### Prerequisites

_

### Installing

_


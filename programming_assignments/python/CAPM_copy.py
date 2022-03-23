#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:12:54 2022

@author: lucywang
"""

# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252

# read in the market data
data = pd.read_csv('capm_market_data.csv')

Look at some records  
SPY is an ETF for the S&P 500 (the "stock market")  
AAPL is Apple  
The values are closing prices, adjusted for splits and dividends

data.head()

Drop the date column

data2 = data.drop(columns=['date'])

Compute daily returns (percentage changes in price) for SPY, AAPL  
Be sure to drop the first row of NaN  
Hint: pandas has functions to easily do this

returns = data2.pct_change()[1:]#.reset_index(drop=True)
returns

#### 1. (1 PT) Print the first 5 rows of returns

returns.head()
spy_returns = returns['spy_adj_close']
aapl_returns = returns['aapl_adj_close']

Save AAPL, SPY returns into separate numpy arrays  
#### 2. (1 PT) Print the first five values from the SPY numpy array, and the AAPL numpy array

spy_returns.head(5)
aapl_returns.head(5)

##### Compute the excess returns of AAPL, SPY by simply subtracting the constant *R_f* from the returns.
##### Specifically, for the numpy array containing AAPL returns, subtract *R_f* from each of the returns. Repeat for SPY returns.

NOTE:  
AAPL - *R_f* = excess return of Apple stock  
SPY - *R_f* = excess return of stock market


spy_excess_returns = spy_returns - R_f
aapl_excess_returns = aapl_returns - R_f

#### 3. (1 PT) Print the LAST five excess returns from both AAPL, SPY numpy arrays


spy_excess_returns.tail(5)
aapl_excess_returns.tail(5)

#### 4. (1 PT) Make a scatterplot with SPY excess returns on x-axis, AAPL excess returns on y-axis####
Matplotlib documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

import matplotlib.pyplot as plt
plt.scatter(spy_excess_returns, aapl_excess_returns)
plt.title("spy excess returns vs aapl excess returns")
plt.xlabel("spy excess returns")
plt.ylabel("aapl excess returns")
plt.show()

#### 5. (3 PTS) Use Linear Algebra (matrices) to Compute the Regression Coefficient Estimate, \\(\hat\beta_i\\)

Hint 1: Here is the matrix formula where *x′* denotes transpose of *x*.

\begin{aligned} \hat\beta_i=(x′x)^{−1}x′y \end{aligned} 

Hint 2: consider numpy functions for matrix multiplication, transpose, and inverse. Be sure to review what these operations do, and how they work, if you're a bit rusty.

import numpy as np
from numpy.linalg import inv

#x_org = np.array(spy_excess_returns)
#x_org2 = x_org.reshape(-1,1)
#y_org = np.array(aapl_excess_returns)
#y_org2 = y_org.reshape(-1,1)

#x_trans = np.transpose(x_org2)
#x_calc = np.matmul(x_trans, x_org2) #reshape and transpose 
#x_inv = np.linalg.inv(x_calc)
#x_calc2 = np.matmul(x_inv,x_trans)
#x_calc3 = np.matmul(x_calc2, y_org2)
#x_calc3

x = np.array(spy_excess_returns).reshape(-1,1)
y = np.array(aapl_excess_returns).reshape(-1,1)
    
bi = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), y)

You should have found that the beta estimate is greater than one.  
This means that the risk of AAPL stock, given the data, and according to this particular (flawed) model,  
is higher relative to the risk of the S&P 500.




#### Measuring Beta Sensitivity to Dropping Observations (Jackknifing)

Let's understand how sensitive the beta is to each data point.   
We want to drop each data point (one at a time), compute \\(\hat\beta_i\\) using our formula from above, and save each measurement.

#### 6. (3 PTS) Write a function called `beta_sensitivity()` with these specs:

- take numpy arrays x and y as inputs
- output a list of tuples. each tuple contains (observation row dropped, beta estimate)

Hint: **np.delete(x, i).reshape(-1,1)** will delete observation i from array x, and make it a column vector

def beta_sensitivity(x, y):
    out = []
    nobs = x.shape[0]
    for ix in range(nobs):
        np.delete(x, ix).reshape(-1,1)
        np.delete(y, ix).reshape(-1,1)
    
        bi = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), y)
        out.append((ix, bi[0][0]))
        
    return out

    

#### Call `beta_sensitivity()` and print the first five tuples of output.

beta_sensitivity(spy_excess_returns, aapl_excess_returns)
#spy_excess_returns
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.varmax import VARMAX

from tqdm import tqdm_notebook
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline


def rolling_predictions(df, train_len, horizon, window, period, method):
    
    TOTAL_LEN = train_len + horizon
    
    seasonal_steps = int((window/period))
    
    if method == 'mean':
        pred_mean = []
        
        for i in range(train_len, TOTAL_LEN, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        
        return pred_mean[:horizon]

    elif method == 'last':
        pred_last_value = []
        
        for i in range(train_len, TOTAL_LEN, window):
            last_value = diff[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))

        return pred_last_value[:horizon]
    
    elif method == 'last_season':
        pred_last_season = []
        
        for i in range(train_len, TOTAL_LEN, window):
            last_season = df[:i][-period:].values
            pred_last_season.extend(last_season for _ in range(seasonal_steps))

        pred_last_season = np.array(pred_last_season).reshape(1, -1)
        
        return pred_last_season[0][:horizon]
    
    if method == 'ARIMA':
        pred_ARIMA = []
        
        for i in range(train_len, TOTAL_LEN, window):
            model = SARIMAX(df[:i], order=(4,1,4))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean[-window:]
            pred_ARIMA.extend(oos_pred)
            
        return pred_ARIMA[:horizon]
    


def ljung_box_test(residuals, is_seasonal, period):

    if is_seasonal:
        lb_df = acorr_ljungbox(residuals, period=period)
    else:
        max_lag = min([10, len(residuals)/5])
        
        lb_df = acorr_ljungbox(residuals, np.arange(1, max_lag+1, 1))

    fig, ax = plt.subplots()
    ax.plot(lb_df['lb_pvalue'], 'b-', label='p-values')
    ax.hlines(y=0.05, xmin=1, xmax=len(lb_df), color='black')
    plt.tight_layout()

    if all(pvalue > 0.05 for pvalue in lb_df['lb_pvalue']):
        print('All values are above 0.05. We fail to reject the null hypothesis. The residuals are uncorrelated')
    else:
        print('One p-value is smaller than 0.05')



def ARIMA_gridsearch(endog, min_p, max_p, min_q, max_q, d):
    
    all_p = range(min_p, max_p+1, 1)
    all_q = range(min_q, max_q+1, 1)
    
    all_orders = list(product(all_p, all_q))
    
    print(f'Fitting {len(all_orders)} unique models')
    
    results = []
    
    for order in tqdm_notebook(all_orders):
        try: 
            model = SARIMAX(endog, order=(order[0], d, order[1])).fit()
        except:
            continue
            
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df

def SARIMAX_gridsearch(endog, exog, min_p, max_p, min_q, max_q, min_P, max_P, min_Q, max_Q, d, D, s):
    
    all_p = range(min_p, max_p+1, 1)
    all_q = range(min_q, max_q+1, 1)
    all_P = range(min_P, max_P+1, 1)
    all_Q = range(min_Q, max_Q+1, 1)
    
    all_orders = list(product(all_p, all_q, all_P, all_Q))
    
    print(f'Fitting {len(all_orders)} unique models')
    
    results = []
    
    for order in tqdm_notebook(all_orders):
        try: 
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue
            
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df



def VAR_gridsearch(endog, min_p, max_p):
    
    all_p = range(min_p, max_p +1, 1)
    
    results = []
    
    print(f'Fitting {all_p} unique models')
    
    for p in tqdm_notebook(all_p):
        try:
            model = VARMAX(endog, order=(p, 0)).fit(dips=False)
        except:
            continue
            
        results.append([p, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['p', 'AIC']
    
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df

def rolling_predictions(df, last_train_value, train_len, horizon, window, method):
    
    total_len = train_len + horizon
    
    if method == 'VAR':

        cows_pred_VAR = []
        calves_pred_VAR = []
        
        for i in range(train_len, total_len, window):
            model = VARMAX(df[:i], order=(6,0))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            
            oos_pred_cows = predictions.predicted_mean.iloc[-window:]['cows']
            oos_pred_calves = predictions.predicted_mean.iloc[-window:]['calves']
            
            cows_pred_VAR.extend(oos_pred_cows)
            calves_pred_VAR.extend(oos_pred_calves)
            
        cows_pred_VAR = np.insert(cows_pred_VAR, 0, last_train_value['cows'])
        cows_pred_VAR = cows_pred_VAR.cumsum()
        
        calves_pred_VAR = np.insert(calves_pred_VAR, 0, last_train_value['calves'])
        calves_pred_VAR = calves_pred_VAR.cumsum()
        
        return cows_pred_VAR[:horizon], calves_pred_VAR[:horizon]
    
    elif method == 'last':
        cows_pred_last = []
        calves_pred_last = []
        
        for i in range(train_len, total_len, window):
            
            cows_last = df[:i].iloc[-1]['cows']
            calves_last = df[:i].iloc[-1]['calves']
            
            cows_pred_last.extend(cows_last for _ in range(window))
            calves_pred_last.extend(calves_last for _ in range(window))
        
        cows_pred_last = np.insert(cows_pred_last, 0, last_train_value['cows'])
        cows_pred_last = cows_pred_last.cumsum()
        
        calves_pred_last = np.insert(calves_pred_last, 0, last_train_value['calves'])
        calves_pred_last = calves_pred_last.cumsum()
            
        return cows_pred_last[:horizon], calves_pred_last[:horizon]
    
def VARMA_gridsearch(endog, min_p, max_p, min_q, max_q):
    
    all_p = range(min_p, max_p+1, 1)
    all_q = range(min_q, max_q+1, 1)
    all_orders = list(product(all_p, all_q))
    
    results = []
    
    print(f'Fitting {len(all_orders)} unique models')
    
    for order in tqdm_notebook(all_orders):
        try:
            model = VARMAX(endog, order=order).fit(disp=False)
        except:
            continue
    
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df
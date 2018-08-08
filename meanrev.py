# Implements mean reversion investment strategy

import argparse
import numpy as np
import pandas as pd
import sys


__author__ = 'Erhan Bilal'

def get_args():
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Implements mean reversion investment strategy.')
    # Add arguments
    parser.add_argument(
        '-f', '--file', type=str, help='CSV File with stock prices, one per column.', required=True)
    parser.add_argument(
        '-p', '--pastn', type=int, help='Number of past values to train on (Default 252).', required=False, default=252)
    parser.add_argument(
        '-r', '--rate', type=int, help='Recalculate stock rankings every r steps (Default 63).', required=False, default=63)
    parser.add_argument(
        '-n', '--topn', type=int, help='Top n stocks to combine (Default 50).', required=False, default=50)
    parser.add_argument(
        '-c', '--capital', type=float, help='Initial capital (Default 10k).', required=False, default=10000)
    parser.add_argument(
        '-o', '--order', type=int, help='Order of stock prices: 1 - ascending, 0 - descending (Default 1).', required=False, default=1)
    parser.add_argument(
        '-rr', '--returnrate', type=int, help='Display investment return every rr days (Default 252).', required=False, default=252)   
    parser.add_argument(
        '-sn', '--startn', type=int, help='Start first training after sn days. (Default 0).', required=False, default=0)    
    parser.add_argument(
        '-tc', '--tradecost', type=float, help='The cost of a trade, regardless of number of stocks (Default 0).', required=False, default=0)
    parser.add_argument(
        '-sp', '--spread', type=float, help='Percentage loss of each sell transaction [0,100] (Default 0).', required=False, default=0)   
    parser.add_argument(
        '-v', '--verbose', type=int, help='Display additional information {0,1} (Default 0).', required=False, default=0)        
    parser.add_argument(
        '-d', '--dev', type=int, help='How to calculate stock performance: 0 - deviation from the mean, 1 - deviation from initial value (Default 1).', required=False, default=1)        
        
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    filename = args.file
    pastn = args.pastn
    rate = args.rate
    topn = args.topn
    initcapital = args.capital
    order = args.order
    returnrate = args.returnrate
    tradecost = args.tradecost
    spread = args.spread/100
    verbose = args.verbose
    startn = args.startn
    dev = args.dev
    
    # Return all variable values
    return filename, pastn, rate, topn, initcapital, order, returnrate, tradecost, spread, verbose, startn, dev


def main():
    filename, pastn, rate, topn, initcapital, order, returnrate, tradecost, spread, verbose, startn, dev = get_args()
    # Read stock data into ndarray
    DF = pd.DataFrame.from_csv(filename)
    dates = DF.index.tolist()
    D = DF.as_matrix()
    D = D[startn:,:]
    dates = dates[startn:]
    
    n, m = D.shape
    # Check for option errors
    assert pastn < n, "PASTN should be smaller than the number of historic stock prices."
    assert returnrate < n, "RETURNRATE should be smaller than the number of historic stock prices."    
    assert initcapital > 0, "CAPITAL must be a positive value."
    assert pastn > 0, "PASTN must be a positive value."
    assert rate > 0, "RATE must be a positive value."
    assert topn > 0, "TOPN must be a positive value."
    assert returnrate > 0, "RETURNRATE must be a positive value."
    assert tradecost >= 0, "TRADECOST must be a positive value."
    assert (spread >= 0) & (spread <= 100), "SPREAD has a value between [0,100]."
    assert (verbose == 0) | (verbose == 1), "VERBOSE should be 0 or 1."
    
    print 'Initial capital: $%.2f' % initcapital
    print 'Number of past records to train on: %d' % pastn
    print 'Rate of stock rebalancing: %d' % rate
    print 'Number of top stocks to keep: %d' % topn
    print 'Trade cost: $%.2f' % tradecost
    print 'Stock spread: %.2f%%' % (spread*100)
    sys.stdout.flush()
    
    
    if order == 0:
        D = np.flipud(D)      
        
    capital = initcapital*np.ones([1,])
    PRT = np.zeros([len(range(pastn, n, rate)), m])
    iCapital = slice(pastn-1,n)
    count = 0
    start_idx = 0
    ROI = []
    print "Weighting stocks..."
    for k in range(pastn, n, rate):      
        
        if k <= n-rate:
            X = D[k-pastn:k+rate]
        else:
            X = D[k-pastn:]
        
        iValid = ~np.any(np.isnan(X), axis=0) 
        
        perf = np.empty([m,])
        perf[:] = np.NaN
        if (dev == 0):
            perf[iValid] = get_dev_mean(X[0:pastn,iValid])
        else:
            perf[iValid] = get_dev_abs(X[0:pastn,iValid])
        ind = np.argsort(perf)
        
        if sum(iValid) < topn:
            newtopn = sum(iValid)
        else:
            newtopn = topn
        w = np.ones([newtopn,]) / newtopn
        
        
        if count == 0:
            transaction_cost = newtopn*tradecost
        else:
            transaction_cost = get_cost(PRT[count-1,:], X[pastn-1,:], tradecost, spread)
        
        if verbose == 1:
            print '%s Current capital = $%.2f' % (dates[k], capital[-1])
            print '%s Transaction cost = $%.2f' % (dates[k], transaction_cost)
                          
        # Calculate future capital
        if capital[-1] > transaction_cost:                 
            capital[-1] = capital[-1] - transaction_cost
        else:
            print '%s Transaction costs used up all your capital!' % dates[k]
            sys.exit(0)
           
        portofolio = capital[-1]*w / X[pastn-1,ind[0:newtopn]]
        PRT[count,ind[0:newtopn]] = portofolio
        newcapital = X[pastn:,ind[0:newtopn]].dot(portofolio)
        capital = np.concatenate((capital,newcapital), axis=0)
        
        # Display investment return
        for i in range(start_idx+returnrate, len(capital), returnrate):
            roi = 100*(capital[i] - capital[i-returnrate]) / capital[i-returnrate]
            ROI.append(roi)
            print '%s ROI = %.2f%%' % (dates[pastn+i-1], roi)
            sys.stdout.flush()
            start_idx = i
                    
        count = count + 1
            
    avg_roi = sum(ROI)/len(ROI)  
    std_roi = np.std(ROI)
    snr = avg_roi/std_roi
    print 'Average ROI %.2f%% with %.2f%% standard deviation.' % (avg_roi, std_roi)
    print 'SNR = %.2f' % snr
    print 'Final capital = $%.2f' % capital[-1]
        
    capital_df = pd.DataFrame(capital, index=dates[iCapital], columns=["Capital"])
    capital_df.to_csv('capital_' + filename)
    
    print "Find how your investment changed over time in the file capital_%s" % filename

def get_dev_mean(X):
    dev = (X[-1,:] - np.mean(X, axis=0)) / np.mean(X, axis=0)
    return dev
    
def get_dev_abs(X):
    dev = (X[-1,:] - X[0,:]) / X[0,:]
    return dev
    
def get_cost(prtf, price, tc, sp):
    # Calculate the optimum costs for rebalance
    iValid = ~np.isnan(price)
    
    price = price[iValid]
    prtf = prtf[iValid]
    w2 = np.ones([len(prtf),]) / len(prtf)
    
    tcap = price.dot(prtf)
    w1 = prtf*price/tcap
    
    ntc = sum(np.not_equal(w1, w2))
    ind = np.greater(w1, w2)
    m = sum(ind)
    A = np.empty([m, m])
    b = np.empty([m, ])
    prtf = prtf[ind]
    price = price[ind]
    w1 = w1[ind]
    w2 = w2[ind]

    for i in range(m):
        for j in range(m):
            if i == j:
                A[i,j] = price[i] - sp*w2[i]*price[i]
            else:
                A[i,j] = - sp*w2[j]*price[j]
                
    for i in range(m):
        b[i] = w2[i]*tcap - sp*w2[i]*price.dot(prtf) - ntc*tc

    newprtf = np.linalg.solve(A, b)    
    
    cost = 0
    for i in range(m):
        if newprtf[i] < prtf[i]:
            cost = cost + sp*price[i]*(prtf[i] - newprtf[i])
    cost = cost + ntc*tc
    return cost    
    
if __name__ == '__main__':
    main()

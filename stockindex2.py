# Ranks and combines stocks based on historic prices
from __future__ import division
import argparse
import scipy as sp
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn import metrics

__author__ = 'Erhan Bilal'

def get_args():
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Ranks and combines stocks based on historic prices.')
    # Add arguments
    parser.add_argument(
        '-f', '--file1', type=str, help='CSV File with adjusted stock prices, one per column.', required=True)
    parser.add_argument(
        '-g', '--file2', type=str, help='CSV File with unadjusted stock prices, one per column.', required=True)
    parser.add_argument(
        '-ef', '--msefile', type=str, help='Read pre-calculated stock mse values from mse_*csv file', required=False)
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
        '-b', '--bottom', type=int, help='Choose top or bottom ranked stocks: 0 - top n stocks, 1 - bottom n stocks (Default 0).', required=False, default=0)
    parser.add_argument(
        '-w', '--weight', type=int, help='How to weight the TOPN stocks: 0 - equal weights, 1 - non-equal weights (Default 0).', required=False, default=0)
    parser.add_argument(
        '-s', '--solver', type=int, help='Choose solver: 0 - spectral decomposition, 1 - least squares, 2 - spectral decomposition plus least squares (Default 1).', required=False, default=0)
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

    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    file1 = args.file1
    file2 = args.file2
    pastn = args.pastn
    rate = args.rate
    topn = args.topn
    initcapital = args.capital
    order = args.order
    solver = args.solver
    returnrate = args.returnrate
    weight = args.weight
    tradecost = args.tradecost
    spread = args.spread/100
    verbose = args.verbose
    msefile = args.msefile
    startn = args.startn
    bottom = args.bottom

    # Return all variable values
    return file1, file2, pastn, rate, topn, initcapital, order, solver, returnrate, weight, tradecost, spread, verbose, msefile, startn, bottom


def main():
    file1, file2, pastn, rate, topn, initcapital, order, solver, returnrate, weight, tradecost, spread, verbose, msefile, startn, bottom = get_args()
    # Read stock data into ndarray
    DF = pd.DataFrame.from_csv(file1)
    tickers = DF.columns.values.tolist()
    dates = DF.index.tolist()
    D = DF.as_matrix()
    D = D[startn:,:]
    dates = dates[startn:]
    DF = pd.DataFrame.from_csv(file2)
    D2 = DF.as_matrix()
    D2 = D2[startn:,:]

    n, m = D.shape
    # Check for option errors
    assert pastn < n, "PASTN should be smaller than the number of historic stock prices."
    #assert topn <= m, "TOPN should be smaller than the total number of stocks."
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
        D2 = np.flipud(D2)

    if msefile is None:
        MSE = np.empty([len(range(pastn, n, rate)), m])
        MSE[:] = np.NaN
    else:
        MSE = pd.DataFrame.from_csv(msefile).as_matrix()

    capital = initcapital*np.ones([1,])
    WHT = np.zeros([len(range(pastn, n, rate)), m])
    PRT = np.zeros([len(range(pastn, n, rate)), m])
    weight_dates = []
    iCapital = slice(pastn-1,n)
    count = 0
    start_idx = 0
    ROI = []
    print "Weighting stocks..."
    for k in range(pastn, n, rate):
        weight_dates.append(dates[k])

        if k <= n-rate:
            X = D[k-pastn:k+rate]
            X2 = D2[k-pastn:k+rate]
        else:
            X = D[k-pastn:]
            X2 = D2[k-pastn:]

        iValid = ~np.any(np.isnan(X2), axis=0)

        if msefile is None:
            if solver == 0:
                mse = get_sd_mse(X2[0:pastn,iValid])
                MSE[count,iValid] = mse
                mse = MSE[count,:]
            elif solver == 1:
                mse = get_ls_mse(X2[0:pastn,iValid])
                MSE[count,iValid] = mse
                mse = MSE[count,:]
            elif solver == 2:
                mse = get_sdls_mse(X2[0:pastn,iValid])
                MSE[count,iValid] = mse
                mse = MSE[count,:]
        else:
            mse = MSE[count,:]

        if bottom == 0:
            ind = np.argsort(mse)
        else:
            ind = np.argsort(mse)[::-1]
            ind = ind[len(ind)-sum(iValid):]

        if sum(iValid) < topn:
            newtopn = sum(iValid)
        else:
            newtopn = topn

        if (weight == 1) & (solver == 1):
            mse_ = mse[ind[0:newtopn]]
            iz = np.argwhere(mse_ == 0)
            if len(iz) == newtopn:
                mse_ = np.ones([newtopn,])
            elif iz.size > 0:
                mse_[iz] = min(mse_[np.nonzero(mse_)])
            invmse = 1/mse_
            w = invmse/sum(invmse)
        else:
            w = np.ones([newtopn,]) / newtopn
        WHT[count,ind[0:newtopn]] = w

        if count == 0:
            transaction_cost = newtopn*tradecost
        else:
            transaction_cost = get_cost(PRT[count-1,:], X[pastn-1,:], WHT[count,:], tradecost, spread)

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

    if len(ROI) > 0:
		avg_roi = sum(ROI)/len(ROI)
		std_roi = np.std(ROI)
		snr = avg_roi/std_roi
		print 'Average ROI %.2f%% with %.2f%% standard deviation.' % (avg_roi, std_roi)
		print 'SNR = %.2f' % snr
    print 'Final capital = $%.2f' % capital[-1]

    capital_df = pd.DataFrame(capital, index=dates[iCapital], columns=["Capital"])
    capital_df.to_csv('capital_' + file1)
    weights_df = pd.DataFrame(WHT, index=weight_dates, columns=tickers)
    weights_df.to_csv('weights_' + file1)
    if msefile is None:
        mse_df = pd.DataFrame(MSE, index=weight_dates, columns=tickers)
        mse_df.to_csv('mse_' + file1)

    print "Find how your investment changed over time in the file capital_%s" % file1

def get_cost(prtf, price, w2, tc, sp):
    # Calculate the optimum costs for rebalance
    iValid = ~np.isnan(price)

    price = price[iValid]
    prtf = prtf[iValid]
    w2 = w2[iValid]

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

def get_sd_mse(X):
    # Mean center the data
    X = preprocessing.scale(X, axis=0, with_mean=True, with_std=False)
    n, m = X.shape

    M = sp.spatial.distance.squareform((1/n)*sp.spatial.distance.pdist(np.transpose(X),'sqeuclidean'))
    m = M.max()
    EM = np.exp(M - m)
    #vals, vecs = sp.sparse.linalg.eigsh(EM, k=1, which='LA')
    vals, vecs = np.linalg.eigh(EM)

    if np.median(vecs[:,-1]) < 0:
        mse = -vecs[:,-1]
    else:
        mse = vecs[:,-1]
    return mse


def get_sdls_mse(X):

    mse0 = get_ls_mse(X)
    mse0[np.isnan(mse0)] = 0
    mse0[np.isinf(mse0)] = 0

    n, m = X.shape
    Y = preprocessing.scale(X, axis=0, with_mean=True, with_std=False)

    M = sp.spatial.distance.squareform((1/n)*sp.spatial.distance.pdist(np.transpose(Y),'sqeuclidean'))
    M = M + np.diag(2*mse0)

    EM = np.exp(M/M.max())
    vals, vecs = sp.sparse.linalg.eigsh(EM, k=1, which='LM')
    #vals, vecs = np.linalg.eigh(EM)

    if np.median(vecs[:,-1]) < 0:
        mse = -vecs[:,-1]
    else:
        mse = vecs[:,-1]
    return mse


def get_ls_mse(X):
    # Mean center the data
    Y = preprocessing.scale(X, axis=0, with_mean=True, with_std=False)
    n, m = Y.shape

    A = np.zeros([int(m*(m-1)/2), m], dtype=np.float32)
    b = np.zeros([int(m*(m-1)/2), ], dtype=np.float32)
    k = 0
    for i in range(0,m):
        for j in range(0,i):
            A[k,i] = 1
            A[k,j] = 1
            b[k] = metrics.mean_squared_error(Y[:,i], Y[:,j])
            k = k + 1

    mse, res = sp.optimize.nnls(A, b)
    return mse

def get_lad_mse(X):
    # Mean center the data
    X = preprocessing.scale(X, axis=0, with_mean=True, with_std=False)
    n, m = X.shape
    n = int(0.5*m*(m-1))

    c = np.concatenate((np.zeros([m,], dtype=np.float32),np.ones([n,], dtype=np.float32)), axis=0)

    Y = np.zeros([n,m], dtype=np.int8)
    y = np.zeros([n,], dtype=np.float32)
    k = 0
    for i in range(0,m):
        for j in range(0,i):
            Y[k,i] = 1
            Y[k,j] = 1
            y[k] = metrics.mean_squared_error(X[:,i], X[:,j])
            k = k + 1;

    A = np.zeros([2*n,n+m], dtype=np.int8)
    A[0:n,0:m] = -Y
    A[n:2*n,0:m] = Y
    A[0:n,m:m+n] = -1*np.identity(n,dtype=np.int8)
    A[n:2*n,m:m+n] = -1*np.identity(n,dtype=np.int8)
    b = np.hstack((-y,y))
    bounds = []

    for i in range(0,m):
        bounds.append((0,None))
    for i in range(0,n):
        bounds.append((None,None))

    prm = sp.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, options=dict(bland=False,maxiter=20*(n+m),tol=1e-6))
    if prm.status == 0:
        return prm.x[0:m]
    else:
        print prm.message
        sys.exit(0)

if __name__ == '__main__':
    main()

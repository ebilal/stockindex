# stockindex
Experimenting with index rebalancing algorithms.

SPY or other index funds that follow SP500 index generally use the market capitalization of top ~500 US companies as weights associated with each ticker. Since it's creation, SP500 has been remrkably consistent in its growth, and over the long term consistently beats other more managed funds, especially when taking into account the typical 2-20 fees. 

This project aims to find better weights that improve on the long term performance of SP500. The central idea is that each ticker represents a prediction about how the economy is doing. So the question becomes what is the best way to combine the stocks in order to achive this goal. Since there is no perfect measurement of "how the economy is doing" (lets call this variable m) we'll assume it's unknown. Viewed through the prism of unsupervised ensemble regression, for stock s_i if e_ij = E[s_i - s_j] than we can find e_i = E[s_i - m] by solving minE[sij - s_ -s_j]. The weights for each stock are now w = 1 / s_i.

Interestingly, based on the SP500 historical data, the weights w perform much better when the original uncorrected stock price is considered, without normalizing it. The resulting index will have mostly stocks prices around $50 with some in single and triple digits. The single digit ones provide high percentage growth and volatility, while the triple digit stocks have lower percentage gains but are more stable, at least based on the training data considered.

How to run:
```python
python stockindex2.py -f sp500v6_adj.csv -g sp500v6_unadj.csv -p 126 -r 126 -n 50

python stockindex2.py -h
usage: stockindex2.py [-h] -f FILE1 -g FILE2 [-ef MSEFILE] [-p PASTN]
                      [-r RATE] [-n TOPN] [-c CAPITAL] [-o ORDER] [-b BOTTOM]
                      [-w WEIGHT] [-s SOLVER] [-rr RETURNRATE] [-sn STARTN]
                      [-tc TRADECOST] [-sp SPREAD] [-v VERBOSE]

Ranks and combines stocks based on historic prices.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE1, --file1 FILE1
                        CSV File with adjusted stock prices, one per column.
  -g FILE2, --file2 FILE2
                        CSV File with unadjusted stock prices, one per column.
  -ef MSEFILE, --msefile MSEFILE
                        Read pre-calculated stock mse values from mse_*csv
                        file
  -p PASTN, --pastn PASTN
                        Number of past values to train on (Default 252).
  -r RATE, --rate RATE  Recalculate stock rankings every r steps (Default 63).
  -n TOPN, --topn TOPN  Top n stocks to combine (Default 50).
  -c CAPITAL, --capital CAPITAL
                        Initial capital (Default 10k).
  -o ORDER, --order ORDER
                        Order of stock prices: 1 - ascending, 0 - descending
                        (Default 1).
  -b BOTTOM, --bottom BOTTOM
                        Choose top or bottom ranked stocks: 0 - top n stocks,
                        1 - bottom n stocks (Default 0).
  -w WEIGHT, --weight WEIGHT
                        How to weight the TOPN stocks: 0 - equal weights, 1 -
                        non-equal weights (Default 0).
  -s SOLVER, --solver SOLVER
                        Choose solver: 0 - spectral decomposition, 1 - least
                        squares, 2 - spectral decomposition plus least squares
                        (Default 0).
  -rr RETURNRATE, --returnrate RETURNRATE
                        Display investment return every rr days (Default 252).
  -sn STARTN, --startn STARTN
                        Start first training after sn days. (Default 0).
  -tc TRADECOST, --tradecost TRADECOST
                        The cost of a trade, regardless of number of stocks
                        (Default 0).
  -sp SPREAD, --spread SPREAD
                        Percentage loss of each sell transaction [0,100]
                        (Default 0).
  -v VERBOSE, --verbose VERBOSE
                        Display additional information {0,1} (Default 0).
```

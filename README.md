# stockindex
Experimenting with index rebalancing algorithms.

SPY or other index funds that follow SP500 index generally use the market capitalization of top ~500 US companies as weights associated with each ticker. Since it's creation, SP500 has been remrkably consistent in its growth, and over the long term consistently beats other more managed funds, especially when taking into account the typical 2-20 fees. 

This project aims to find better weights that improve on the long term performance of SP500. The central idea is that each ticker represents a prediction about how the economy is doing. So the question becomes what is the best way to combine the stocks in order to achive this goal. Since there is no perfect measurement of "how the economy is doing" (lets call this variable m) we'll assume it's unknown. Viewed through the prism of unsupervised ensemble regression, for stock s_i if e_ij = E[s_i - s_j] than we can find e_i = E[s_i - m] by solving minE[sij - s_ -s_j]. The weights for each stock are now w = 1 / s_i.

Interestingly, based on the SP500 historical data, the weights w perform much better when the original uncorrected stock price is considered, without normalizing it. The resulting index will have mostly stocks prices around $50 with some in single and triple digits. The single digit ones provide high percentage growth and volatility, while the triple digit stocks have lower percentage gains but are more stable, at least based on the training data considered.

How to run:
```python
stockindex2.py -f sp500v6_adj.csv -g sp500v6_unadj.csv -p 126 -r 126 -n 50
```

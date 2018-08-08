# stockindex
Experimenting with index rebalancing algorithms.

SPY or other index funds that follow SP500 index generally use the market capitalization of top ~500 US companies as weights associated with each ticker. Since it's creation, SP500 has been remrkably consistent in its growth, and over the long term consistently beats other more managed funds, especially when taking into account the typical 2-20 fees. 

This project aims to find better weights that improve on the long term performance of SP500. The central idea is that each ticker represents a prediction about how the economy is doing. So the question becomes what is the best way to combine the stocks in order to achive this goal. Since there is no perfect measurement of "how the economy is doing" we'll assume it's unknown. Viewed through the prism of unsupervised ensemble regression, if $e_ij $

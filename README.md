# Monte Carlo Options Pricer
This application prices options for a given commodity.

## Everything to know about options
### What is an option?
An **option** is a contract that gives the holder the **right**, but not the **obligation**, to buy or sell something (usually a stock) at a specified price (called the **strike price**) before or on a certain date. **Call options** represent the right to buy a stock at the strike price, and **put options** are the right to sell at the strike price. Options are useful as they provide a way for risk management and speculation, and users are willing to pay for this flexibility.

### Where do options come from?
Options are provided by **option writers** (also called **issuers**) who make these financial products available for trading.

Market Makers:
- professional traders or firms that provide liquidity to the options market by continuously offering to buy and sell options
- they write options, set bid-ask spreads, and profit from the difference
- they are typically connected directly to the exchange and ensure that options are available for trading

Institutions:
- large financial institutions, such as investment banks or hedge funds, often write options as part of their trading strategies
- they may write options to hedge their positions, speculate on market movements, or generate income

Individual Investors:
- sophisticated retail investors can also write options (e.g., sell covered calls or cash-secured puts).
- these investors must meet margin requirements and have appropriate approval from their brokerage.

Regular users don't interact directly with the writers; they access options through **organized markets** or **exchanges**.

### Why is option pricing important?
It's important to price an option correctly, such that it is competitive with the market. If an option is valued too low compared to the fair value, it becomes a no-brainer for a buyer and is against the seller's interests. Similarly, if it is valued too high, it becomes disadvantageous for the buyer and good for the seller. If one side is able to assess the fair value of an option better than others, they gain a competitive edge as they can easily determine if it is profitable to buy or sell a specific option.

### How will we price options?
Pricing an option is a difficult problem to tackle, and there is no silver-bullet to know for certainty what the fair value is. There are a few factors that we can consider when attempting to price an option:
- How much the underlying stock is worth today
- How uncertain the stock's future is (volatility)
- How much time is left until the option expires
- Interest rates, because money today is worth more than money tomorrow

## Building the pricer
Stocks don’t move predictably; they’re random. However, randomness isn’t chaotic—it follows statistical rules. Think of geometric Brownian motion (GBM) as the random process that describes stock price movements:

```math
dSt​=μSt​dt+σSt​dWt
```

Translation: Stock price (St) changes over time (t) due to:
Drift (μ): A steady trend up or down.
Volatility (σ): Randomness scaled by how "jumpy" the stock is.
Random noise (dWt): A fancy term for randomness modeled as "normal distribution" steps.

In finance, we simplify the world to assume that every stock grows, on average, at the risk-free rate (like a guaranteed bank interest rate). This removes "preferences" (like greed vs. fear) and makes pricing consistent.

Instead of guessing what the stock will actually do, we assume it grows at the risk-free rate and price the option accordingly.

This gives us the equation to model future stock prices under this assumption:
ST=S0⋅e(r−σ22)T+σTZ
ST​=S0​⋅e(r−2σ2​)T+σT
​Z

Where Z∼N(0,1)Z∼N(0,1) (a random number from a standard normal distribution).

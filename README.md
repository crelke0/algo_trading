# Quant Research Playground #

A personal quantitative research framework for prototyping, testing, and validating algorithmic trading strategies, with a focus on statistical arbitrage and machine learning–driven signals.

This project is built for research rather than live trading


## Philosophy ##

- Rapid prototyping of new alpha ideas
- Thorough validation and comparison of strategies
- Clear separation between strategy logic and backtesting engine
- Simple strategy composability


## What I've been exploring ##

- ML statistical arbitrage
- Anomaly detection (and possible signal generation) via SVDD
- Mean reversion & residual trading
- ML-based signal generation
- Monte-carlo permutation testing


## What's coming ##

- Hedging to minimize short positions
- Portfolio generation through analyzing and composing strategies


## Project layout ##

Strategies are in strategies.py. Some strategies have to be trained, in which case they have a "maker" that returns the strategy. E.g. "make_MLP_stat_arb" returns a strategy after training on the provided data.

The engine and diagnostic functions are in main.py

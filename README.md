Currently In Development

### About

This project aims to allow for creating RL trading agents on OpenBB sourced datasets.

It aims to create a more Gymnasium Native approach to Tensortrade's modular design.

The focus will be on writing customized action and reward schemes and integration with OpenBB's portfolio management system.

The goal is to be able to

- quickly train on data amalgamated by OpenBB
- See how agents perform through Alpaca Paper Trading
- Explore Model-Based Learning Methods (https://arxiv.org/abs/2010.02193)

Basic Usage

1. Create a DF from OpenBB
2. Create a TradingEnvironment from DF
3. Run your RL algo
4. View results

Note: Gymnasium not fully compatible wtih SB3 yet

### Todo

If you want to contribute, here are areas of improvement

1. Trading Environment

- Note: Validate Gymnasium Custom Wrappers over Action/Reward/Observer classes

* Actions & Positions should be integrated into an action scheme
  - Action Scheme gets env
* Portfolio Management handled by
  - Portfolio Engine ?
    - https://docs.openbb.co/sdk/guides/intros/portfolio
* Actions
  - Action Schemes
    - Generic Scheme
    - Simple Actions
    - Contious Actions w Portfolio Reallocation
    - Allow for more trading pairs
* Observation Space
  - Specific to DF's but want generators
    - DF should be treated as a generator
    - Issues
      - Observation Window
  - Keep within environment, allow users to pass custom observation spaces
* Rewards
  - Reward Scheme
    - Want to be able to experiment with differing types of rewards, e.g. trying Sharpe's ratio and so on
    - Calculate a simple return and maximize it based of AnyTrading

2. OpenBB Interface

   - Training data as a dataframe
   - Training data as a socket

3. Create generic components

   - Simulators
     - Alpaca Paper Trading
     - MT5Gym Simulator can be adapted

4. Live Trading

- Integrate MTGym, https://github.com/AminHP/gym-mtsim

5. Other Issues

- np.float64 vs np.float32 for computation speed
- self.current_tick vs self.clock redundancy!
- Quantstats is broken
- Agent training broken

6. Integrate Optuna for hyperparamter finding

- Allow

### Citations

Adapted from the following resources

- https://github.com/AminHP/gym-anytrading
- https://github.com/tensortrade-org/tensortrade
- https://github.com/AminHP/gym-mtsim

Big thanks to https://github.com/AminHP, OpenBB, and TensorTrade contributers!

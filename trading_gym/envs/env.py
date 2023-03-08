import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from abc import ABC

# rendering
import matplotlib.pyplot as plt
from enum import Enum


# Defines default set of possible discrete actions
class Actions(Enum):
    Sell = 0
    Buy = 1


# Defines default set of discrete positions (e.g can buy or sell, short or long)
class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class RewardScheme(ABC):
    def __init__(self):
        pass

    def calculate_reward(self, action, env):
        raise NotImplementedError()

    def update_profit(self, action, env):
        raise NotImplementedError()

    def max_possible_profit(self, env):
        raise NotImplementedError()


# class SimpleProfitForex(RewardScheme):
#     def __init__(self):
#         pass


class SimpleProfitStocks(RewardScheme):
    def __init__(
        self, trade_fee_bid_percent: float = 0.01, trade_fee_ask_percent: float = 0.005
    ):
        # reward dictated by environment fees
        self.trade_fee_bid_percent = trade_fee_bid_percent  # unit
        self.trade_fee_ask_percent = trade_fee_ask_percent  # unit

    def calculate_reward(self, action, env):
        # returns reward for current step!
        step_reward = 0

        if (action == Actions.Buy.value and env._position == Positions.Short) or (
            action == Actions.Sell.value and env._position == Positions.Long
        ):
            price_diff = (
                env.prices[env._current_tick] - env.prices[env._last_trade_tick]
            )
            # in stock environemnt this would mean you've bought and sold.
            if env._position == Positions.Long:
                step_reward += price_diff

        # return reward for current step
        return step_reward

    def update_profit(self, action, env):
        # Trade occurs IF
        # *1. If the current action is to buy, and position is short
        # *2. If the current action was to sell, but position is long
        if (
            (action == Actions.Buy.value and env._position == Positions.Short)
            or (action == Actions.Sell.value and env._position == Positions.Long)
            or env._done
        ):
            current_price = env.prices[env._current_tick]
            last_trade_price = env.prices[env._last_trade_tick]

            # update if long
            if env._position == Positions.Long:
                shares = (
                    env._total_profit * (1 - self.trade_fee_ask_percent)
                ) / last_trade_price
                env._total_profit = (
                    shares * (1 - self.trade_fee_bid_percent)
                ) * current_price

            # update env's new position and history as we made a trade
            env._position = env._position.opposite()
            env._last_trade_tick = env._current_tick

    def max_possible_profit(self, env):
        current_tick = env._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.0

        while current_tick <= env._end_tick:
            position = None
            if env.prices[current_tick] < env.prices[current_tick - 1]:
                while (
                    current_tick <= env._end_tick
                    and env.prices[current_tick] < env.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Short
            else:
                while (
                    current_tick <= env._end_tick
                    and env.prices[current_tick] >= env.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = env.prices[current_tick - 1]
                last_trade_price = env.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit


class DFTradingEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 5,
        frame_bound: tuple = None,
        reward_strategy: RewardScheme = SimpleProfitStocks(),
        # render_mode: str = "human",
    ):
        # initialize attributes
        self.df = df
        self.window_size = window_size

        # how we calculate and assign rewards
        self.reward_strategy = reward_strategy
        # self.action_scheme = action_scheme

        # specify start and end of dataframe
        self.frame_bound = frame_bound if frame_bound else (self.window_size, len(df))

        #! wrong as a consequence of below
        assert (
            self.frame_bound[0] >= self.window_size
        )  # "Starting boundary must be greater than the required window size"
        self.prices, self.signal_features = self._process_data()

        # * Define Action and Observation Spaces
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # enscapsulate into action scheme

        # *
        # self.action_scheme = ActionScheme
        # self.actions = Actions
        # self.positions = Positions
        # *

        # self.action_space = spaces.Discrete(len(self.actions))
        self.action_space = spaces.Discrete(len(Actions))

        # the env's observation shape and space
        self.shape = (self.window_size, self.signal_features.shape[1])

        #! NOTE: Dict space incompatible with certain policies
        # used a dictionary to allow for cutomizations
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(
        #             low=-np.inf,
        #             high=np.inf,
        #             shape=self.shape,
        #             dtype=np.float64,
        #         ),
        #     }
        # )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.shape,
            dtype=np.float64,
        )

        # * Remove unless using more render modes
        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        # """
        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        # """
        # # * Render Options
        # self.draw_window = None
        # self.clock = None
        # *
        #! Perhaps makes more sense in a or observer?
        # *  episodic information
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = False
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._first_rendering = None
        self.history = None
        self._total_reward = 0.0
        self._total_profit = 1.0

    # * Overwrite for custom processing
    def _process_data(self):
        #! TODO add logging
        # print("Warning: Using default processing function!")

        # take closing prices
        prices = self.df.loc[:, "Close"].to_numpy()

        #! Shouldn't this be + window size,
        #! and we start at the frame_bound + window_size position
        prices[self.frame_bound[0] - self.window_size]

        # TODO: Lookahead bias check
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    # * Private methods to translate env state for step/reset-> observation
    def _get_obs(self):
        # * Dict space
        # return {
        #     "agent": self.signal_features[
        #         (self._current_tick - self.window_size + 1) : self._current_tick + 1
        #     ]
        # }
        # * Box space
        return self.signal_features[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]

    # * Private methods to compute auxillary metrics for step/reset e.g.
    def _get_info(self):
        return {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
            "position": self._position.value,
        }

    def _update_history(self, info):
        # init history dictionary
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        # update history dictionary
        for key, value in info.items():
            self.history[key].append(value)

    def reset(self, seed=None):
        # We need the following line to seed self.np
        super().reset(seed=seed)

        #! Maybe move into get_info, these could be considered auxilary metrics?
        # reset episodic metrics
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.0
        self._total_profit = 1.0
        self._first_rendering = True
        self.history = {}

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        # move to next time step
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._done = True

        # get reward for action
        reward = self.reward_strategy.calculate_reward(action, self)
        self._total_reward += reward

        # update total profit and position if relevant
        self.reward_strategy.update_profit(action, self)

        #! Move into info if possible to standardize
        # keep track of all positions
        self._position_history.append(self._position)

        observation = self._get_obs()
        info = self._get_info()

        # update our auxillary metrics history
        self._update_history(info)

        # * want each action to be rendered, or the price history to be iteratively rendered
        # if self.render_mode == "human":
        #     self._render_frame()
        # try:
        #     return observation, reward, self._done, info
        # except ValueError:
        return observation, reward, self._done, None, info

    def render(self, mode="human"):
        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = "red"
            elif position == Positions.Long:
                color = "green"
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            + " ~ "
            + "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def render_all(self, mode="human"):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], "ro")
        plt.plot(long_ticks, self.prices[long_ticks], "go")

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            + " ~ "
            + "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()


# if __name__ == "__main__":
# test environment
# env = gym.make("df-trader-v0")

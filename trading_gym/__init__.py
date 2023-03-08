from gymnasium.envs.registration import register

from .envs import env
from . import datasets


register(
    id="df-trader-v0",
    entry_point=env.DFTradingEnvironment,
    kwargs={
        "df": datasets.STOCKS_GOOGL,
        "window_size": 10,
        "frame_bound": (10, 300),
    },
)

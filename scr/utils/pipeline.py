import typing as tp

import numpy as np
import pydantic
import scr.utils.decode
import scr.utils.metric
import sklearn
import sklearn.linear_model
from pydantic import BaseModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Pipeline(BaseModel):
    model: tp.Literal["RidgeCV", "RidgeClassifierCV"]
    alphas: tuple = (-1, 10)
    alpha_per_target: bool = False
    eval_metric: tp.Literal["correlate", "balanced_accuracy"]
    decode_function_name: str
    _pipeline: tp.Any
    _eval_function: tp.Callable
    _decode_function: tp.Callable
    model_config = pydantic.ConfigDict(extra="forbid")

    def model_post_init(self, log__: tp.Any) -> None:
        self._pipeline = self._load_pipeline()
        self._eval_function = getattr(scr.utils.metric, self.eval_metric)
        self._decode_function = getattr(scr.utils.decode, self.decode_function_name)

    def _load_pipeline(self) -> sklearn.linear_model:
        alpha_list = np.logspace(*self.alphas)
        if self.model == "RidgeCV":
            model = sklearn.linear_model.RidgeCV(
                alphas=alpha_list, alpha_per_target=self.alpha_per_target
            )
            assert self.eval_metric in [
                "correlate",
            ], f"RidgeCV only supports correlate, current metric is {self.eval_metric}"
        elif self.model == "RidgeClassifierCV":
            model = sklearn.linear_model.RidgeClassifierCV(alphas=alpha_list)
            assert self.eval_metric in [
                "balanced_accuracy",
            ], f"RidgeClassifierCV only supports balanced_accuracy, current metric is {self.eval_metric}"
        else:
            raise ValueError(f"model {self.model} not supported")

        pipeline = make_pipeline(StandardScaler(), model)

        return pipeline

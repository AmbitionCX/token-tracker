"""Sklearn / XGBoost — external features only (Table 1 row 1)."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class SklearnBaselines:
    @staticmethod
    def logistic_regression(**kwargs: Any) -> LogisticRegression:
        return LogisticRegression(max_iter=1000, **kwargs)

    @staticmethod
    def random_forest(**kwargs: Any) -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=100, **kwargs)

    @staticmethod
    def xgboost(**kwargs: Any) -> xgb.XGBClassifier:
        if "objective" not in kwargs:
            kwargs["objective"] = "binary:logistic"
        return xgb.XGBClassifier(**kwargs)


TraditionalBaselines = SklearnBaselines

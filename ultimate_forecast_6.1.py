# -*- coding: utf-8 -*-
"""V6.1 material demand forecasting system.

All V6.1 implementation code lives in this single file: configuration,
data analysis, forecasting methods, model selection, reporting, and the CLI
entry point.
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
np.random.seed(42)
logger = logging.getLogger(__name__)
@dataclass(frozen=True)
class AnalyzerConfig:
    sparse_threshold: float = 0.8
    intermittent_threshold: float = 0.25
    cv_threshold: float = 0.7
    trend_pvalue_threshold: float = 0.05
    trend_r2_threshold: float = 0.1
    seasonal_min_length: int = 12


@dataclass(frozen=True)
class IntervalConfig:
    min_interval: int = 13
    max_interval: int = 26
    validation_size: int = 5


@dataclass(frozen=True)
class RuntimeConfig:
    test_periods: int = 5
    random_seed: int = 42
    progress_every: int = 100


BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = (
    BASE_DIR / "ready_data.csv",
    BASE_DIR.parent / "V5" / "ready_data.csv",
    BASE_DIR.parent / "forecast_V6" / "ready_data.csv",
)
V6_RESULT_CANDIDATES = (
    BASE_DIR / "ultimate_prediction_results_v6.csv",
    BASE_DIR.parent / "forecast_V6" / "ultimate_prediction_results_v6.csv",
)

LEGACY_METHODS = {
    "moving_average",
    "ma_3",
    "ma_5",
    "optimized_ses",
    "optimized_des",
    "seasonal_naive",
    "tsb_opt",
    "median_5",
    "interval_based",
}

NEW_V6_METHODS = {
    "croston",
    "adida",
    "optimized_hw",
    "theta",
    "weighted_ma_5",
    "winsorized",
}

TEMPORARILY_REMOVED_METHODS = {
    "sba",
    "naive_drift",
    "damped_trend",
}


class DataAnalyzer:

    CONFIG = AnalyzerConfig()
    THRESHOLD_SPARSE = CONFIG.sparse_threshold
    THRESHOLD_INTERMITTENT = CONFIG.intermittent_threshold
    THRESHOLD_CV = CONFIG.cv_threshold
    TREND_PVALUE_THRESHOLD = CONFIG.trend_pvalue_threshold
    TREND_R2_THRESHOLD = CONFIG.trend_r2_threshold

    def __init__(self, series):
        self.series = np.array(series, dtype=float)
        self.non_zero = self.series[self.series > 0]
        self.analysis = self._analyze()

    def _analyze(self):
        s = self.series
        n = len(s)

        if n == 0 or np.sum(s) == 0:
            return {
                'zero_ratio': 1.0 if n > 0 else 1.0,
                'cv': 0, 'skewness': 0, 'kurtosis': 0, 'volatility': 0,
                'acf_1': 0, 'acf_3': 0, 'acf_5': 0, 'acf_values': [],
                'trend_direction': 'none', 'trend_pvalue': 1.0, 'trend_r2': 0.0,
                'seasonal_flag': False, 'seasonal_period': 0, 'seasonal_acf_threshold': 0.0,
                'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'non_zero_count': 0, 'total_demand': 0.0,
                'recent_trend': 0, 'demand_size_variability': 0,
                'stability': 1.0, 'quantile_25': 0.0, 'quantile_75': 0.0,
            }

        non_zero = self.non_zero

        zero_ratio = np.sum(s == 0) / n
        cv = np.std(non_zero) / np.mean(non_zero) if len(non_zero) > 0 and np.mean(non_zero) > 0 else 0

        if len(non_zero) >= 3 and np.std(non_zero) > 1e-10:
            try:
                skewness = stats.skew(non_zero)
                kurtosis = stats.kurtosis(non_zero)
            except Exception as exc:
                logger.debug("Distribution-shape analysis failed: %s", exc)
                skewness = 0
                kurtosis = 0
        else:
            skewness = 0
            kurtosis = 0

        diff = np.diff(s)
        volatility = np.std(diff) / np.mean(np.abs(s)) if np.mean(np.abs(s)) > 0 else 0

        acf_values = [self._autocorr(s, lag) for lag in range(1, min(13, n))]

        trend_direction, trend_pvalue, trend_r2 = self._detect_trend(s)
        seasonal_flag, seasonal_period, seasonal_acf_threshold = self._detect_seasonality(s)

        recent_trend = self._recent_trend(s)
        stability = self._compute_stability(s)

        return {
            'zero_ratio': zero_ratio,
            'cv': cv,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'volatility': volatility,
            'acf_1': acf_values[0] if len(acf_values) > 0 else 0,
            'acf_3': acf_values[2] if len(acf_values) > 2 else 0,
            'acf_5': acf_values[4] if len(acf_values) > 4 else 0,
            'acf_values': acf_values,
            'trend_direction': trend_direction,
            'trend_pvalue': trend_pvalue,
            'trend_r2': trend_r2,
            'seasonal_flag': seasonal_flag,
            'seasonal_period': seasonal_period,
            'seasonal_acf_threshold': seasonal_acf_threshold,
            'mean': np.mean(s),
            'std': np.std(s),
            'median': np.median(s),
            'non_zero_count': len(non_zero),
            'total_demand': np.sum(s),
            'recent_trend': recent_trend,
            'demand_size_variability': np.std(non_zero) if len(non_zero) > 1 else 0,
            'stability': stability,
            'quantile_25': np.percentile(s, 25),
            'quantile_75': np.percentile(s, 75),
        }

    def _autocorr(self, s, lag):
        if len(s) <= lag or len(s) - lag < 2:
            return 0
        left = s[:-lag]
        right = s[lag:]
        if np.std(left) < 1e-10 or np.std(right) < 1e-10:
            return 0
        try:
            corr = np.corrcoef(left, right)[0, 1]
            return corr if np.isfinite(corr) else 0
        except Exception as exc:
            logger.debug("Autocorrelation failed for lag %s: %s", lag, exc)
            return 0

    def _detect_trend(self, s):
        if len(s) < 3:
            return 'none', 1.0, 0.0
        try:
            x = np.arange(len(s))
            slope, _, r_value, p_value, _ = stats.linregress(x, s)
            r2 = r_value ** 2

            is_significant = p_value < self.TREND_PVALUE_THRESHOLD and r2 > self.TREND_R2_THRESHOLD
            if is_significant:
                direction = 'increasing' if slope > 0 else 'decreasing'
            else:
                direction = 'none'

            return direction, p_value, r2
        except Exception as exc:
            logger.debug("Trend detection failed: %s", exc)
            return 'none', 1.0, 0.0

    def _detect_seasonality(self, s):
        if len(s) < self.CONFIG.seasonal_min_length:
            return False, 0, 0.0

        try:
            n = len(s)
            acf_threshold = 1.96 / np.sqrt(n)

            acf_lag12 = self._autocorr(s, self.CONFIG.seasonal_min_length)
            acf_lag4 = self._autocorr(s, 4) if n > 4 else 0
            acf_lag3 = self._autocorr(s, 3) if n > 3 else 0

            if acf_lag12 > acf_threshold:
                return True, self.CONFIG.seasonal_min_length, acf_threshold
            elif acf_lag4 > acf_threshold or acf_lag3 > acf_threshold:
                period = 4 if acf_lag4 >= acf_lag3 else 3
                return True, period, acf_threshold

            return False, 0, acf_threshold
        except Exception as exc:
            logger.debug("Seasonality detection failed: %s", exc)
            return False, 0, 0.0

    def _recent_trend(self, s):
        if len(s) < 10:
            return 0
        recent = s[-10:]
        x = np.arange(len(recent))
        try:
            slope, _, _, _, _ = stats.linregress(x, recent)
            return slope
        except Exception as exc:
            logger.debug("Recent-trend detection failed: %s", exc)
            return 0

    def _compute_stability(self, s):
        if len(s) < 5:
            return 1.0
        recent = s[-5:]
        older = s[-10:-5] if len(s) >= 10 else s[:-5]
        if len(older) == 0:
            return 1.0
        return np.abs(np.mean(recent) - np.mean(older)) / (np.mean(np.abs(s)) + 1e-10)

    def get_pattern_type(self):
        a = self.analysis

        if np.sum(self.series) == 0:
            return 'all_zero'

        if a['seasonal_flag']:
            return 'seasonal'

        if a['trend_direction'] != 'none':
            return 'trending'

        if a['zero_ratio'] >= self.THRESHOLD_SPARSE:
            return 'sparse'

        is_high_intermittent = a['zero_ratio'] >= self.THRESHOLD_INTERMITTENT
        is_high_volatility = a['cv'] >= self.THRESHOLD_CV

        if is_high_intermittent and is_high_volatility:
            return 'lumpy'
        elif is_high_intermittent and not is_high_volatility:
            return 'intermittent'
        elif not is_high_intermittent and is_high_volatility:
            return 'erratic'
        else:
            return 'stable'


class OptimalIntervalFinder:

    CONFIG = IntervalConfig()
    MIN_INTERVAL = CONFIG.min_interval
    MAX_INTERVAL = CONFIG.max_interval
    VAL_SIZE = CONFIG.validation_size  # internal validation size, separate from final test

    def __init__(self, series):
        self.series = np.array(series, dtype=float)
        self.total_length = len(self.series)
        self.optimal_interval = None
        self.optimal_mase = float('inf')
        self.interval_scores = {}
        self._find_optimal_interval()

    def _find_optimal_interval(self):
        min_required = self.MIN_INTERVAL + self.VAL_SIZE
        if self.total_length < min_required:
            self.optimal_interval = max(self.MIN_INTERVAL, self.total_length - self.VAL_SIZE)
            return

        # Use the end of the series as a validation set (separate from final test set
        # which is handled by AdaptiveModelSelector)
        for interval_len in range(self.MIN_INTERVAL, self.MAX_INTERVAL + 1):
            if interval_len + self.VAL_SIZE > self.total_length:
                continue

            val_end = self.total_length
            val_start = max(0, val_end - self.VAL_SIZE)
            train_end = val_start
            train_start = max(0, train_end - interval_len)

            train_data = self.series[train_start:train_end]
            val_data = self.series[val_end - self.VAL_SIZE:val_end]

            if len(train_data) < self.MIN_INTERVAL:
                continue

            mase = self._evaluate_interval(train_data, val_data)
            self.interval_scores[interval_len] = mase

            if mase < self.optimal_mase:
                self.optimal_mase = mase
                self.optimal_interval = interval_len

        if self.optimal_interval is None:
            self.optimal_interval = self.MIN_INTERVAL

    def _evaluate_interval(self, train, test):
        train = np.array(train, dtype=float)
        test = np.array(test, dtype=float)

        if len(train) < 3:
            return float('inf')

        if len(train) >= 5:
            forecast = np.mean(train[-5:]) * len(test)
        else:
            forecast = np.mean(train) * len(test)

        actual = np.sum(test)
        mae = abs(forecast - actual) / len(test) if len(test) > 0 else 0

        naive_errors = np.abs(np.diff(train))
        if len(naive_errors) > 0 and np.mean(naive_errors) > 1e-10:
            mase = mae / np.mean(naive_errors)
        else:
            mase = mae if mae > 0 else 0.0

        return mase

    def get_optimal_data(self):
        if self.optimal_interval is None:
            self.optimal_interval = self.MIN_INTERVAL

        start_idx = max(0, self.total_length - self.optimal_interval)
        return self.series[start_idx:], self.optimal_interval


class AdaptiveDataAnalyzer:

    def __init__(self, series):
        self.series = np.array(series, dtype=float)
        self.n = len(self.series)

        self.full_analyzer = DataAnalyzer(self.series)
        self.full_pattern = self.full_analyzer.get_pattern_type()
        self.full_analysis = self.full_analyzer.analysis

        interval_finder = OptimalIntervalFinder(self.series)
        self.optimal_interval = interval_finder.optimal_interval
        self.optimal_interval_mase = interval_finder.optimal_mase
        optimal_data, _ = interval_finder.get_optimal_data()

        self.optimal_analyzer = DataAnalyzer(optimal_data)
        self.optimal_pattern = self.optimal_analyzer.get_pattern_type()
        self.optimal_analysis = self.optimal_analyzer.analysis

        self.analysis, self.pattern = self._resolve_pattern()

        self.pattern_shift_detected = (self.full_pattern != self.optimal_pattern)
        self.drift_info = self._compute_drift_info()

    def _resolve_pattern(self):
        return self.optimal_analysis, self.optimal_pattern

    def _compute_drift_info(self):
        return {
            'full_pattern': self.full_pattern,
            'optimal_pattern': self.optimal_pattern,
            'optimal_interval': self.optimal_interval,
            'optimal_interval_mase': self.optimal_interval_mase,
            'pattern_shift': self.pattern_shift_detected,
            'full_zero_ratio': self.full_analysis['zero_ratio'],
            'optimal_zero_ratio': self.optimal_analysis['zero_ratio'],
            'full_cv': self.full_analysis['cv'],
            'optimal_cv': self.optimal_analysis['cv'],
            'full_trend': self.full_analysis['trend_direction'],
            'optimal_trend': self.optimal_analysis['trend_direction'],
        }


class UltimateForecastMethods:

    @staticmethod
    def _mean_or_zero(series):
        return float(np.mean(series)) if len(series) > 0 else 0.0

    @staticmethod
    def moving_average(series, horizon=5, window=5):
        series = np.array(series, dtype=float)
        if len(series) < window:
            avg = UltimateForecastMethods._mean_or_zero(series)
        else:
            avg = np.mean(series[-window:])
        return avg * horizon

    @staticmethod
    def optimized_ses(series, horizon=5):
        series = np.array(series, dtype=float)
        if len(series) < 3:
            return UltimateForecastMethods._mean_or_zero(series) * horizon

        def objective(alpha):
            if alpha <= 0.01 or alpha >= 0.99:
                return 1e10
            result = series[0]
            sse = 0
            for val in series[1:]:
                error = val - result
                sse += error ** 2
                result = alpha * val + (1 - alpha) * result
            return sse

        try:
            result = minimize(objective, x0=[0.3], bounds=[(0.01, 0.99)], method='L-BFGS-B')
            best_alpha = result.x[0]
        except Exception as exc:
            logger.debug("SES optimization failed: %s", exc)
            best_alpha = 0.3

        result = series[0]
        for val in series[1:]:
            result = best_alpha * val + (1 - best_alpha) * result

        return result * horizon

    @staticmethod
    def optimized_des(series, horizon=5):
        series = np.array(series, dtype=float)
        if len(series) < 4:
            return UltimateForecastMethods.optimized_ses(series, horizon)

        def objective(params):
            alpha, beta = params
            if alpha <= 0.01 or alpha >= 0.99 or beta <= 0 or beta >= 0.5:
                return 1e10

            level = series[0]
            trend = series[1] - series[0]
            sse = 0

            for i in range(1, len(series)):
                error = series[i] - (level + trend)
                sse += error ** 2
                new_level = alpha * series[i] + (1 - alpha) * (level + trend)
                trend = beta * (new_level - level) + (1 - beta) * trend
                level = new_level

            return sse

        try:
            result = minimize(objective, x0=[0.3, 0.1], bounds=[(0.01, 0.99), (0.001, 0.3)], method='L-BFGS-B')
            best_alpha, best_beta = result.x
        except Exception as exc:
            logger.debug("DES optimization failed: %s", exc)
            best_alpha, best_beta = 0.3, 0.1

        level = series[0]
        trend = series[1] - series[0]

        for i in range(1, len(series)):
            new_level = best_alpha * series[i] + (1 - best_alpha) * (level + trend)
            trend = best_beta * (new_level - level) + (1 - best_beta) * trend
            level = new_level

        total_forecast = 0
        for h in range(1, horizon + 1):
            total_forecast += max(0, level + h * trend)

        return total_forecast

    @staticmethod
    def seasonal_naive(series, horizon=5, season_length=12):
        series = np.array(series, dtype=float)
        if len(series) < season_length:
            return series[-1] * horizon if len(series) > 0 else 0

        total_forecast = 0
        for h in range(1, horizon + 1):
            idx = len(series) - season_length + (h - 1) % season_length
            if idx >= 0:
                total_forecast += series[idx]

        return total_forecast

    @staticmethod
    def sba(series, horizon=5, alpha=0.1):
        series = np.array(series, dtype=float)
        if len(series) == 0:
            return 0

        non_zero_indices = np.where(series > 0)[0]
        if len(non_zero_indices) == 0:
            return 0

        non_zero_values = series[non_zero_indices]

        if len(non_zero_indices) > 1:
            intervals = np.diff(non_zero_indices)
            mean_interval = np.mean(intervals)
        else:
            mean_interval = len(series)

        level = non_zero_values[0]
        for i in range(1, len(non_zero_values)):
            level = alpha * non_zero_values[i] + (1 - alpha) * level

        if mean_interval > 0:
            expected_demands = horizon / mean_interval
        else:
            expected_demands = horizon

        correction_factor = (1 - alpha / 2)

        total_demand = expected_demands * level * correction_factor

        return max(0, total_demand)

    @staticmethod
    def tsb_opt(series, horizon=5):
        series = np.array(series, dtype=float)
        n = len(series)
        if n == 0:
            return 0

        def objective(params):
            alpha, beta = params
            if alpha <= 0.01 or alpha >= 0.99 or beta <= 0.01 or beta >= 0.99:
                return 1e10

            p = 1.0 if series[0] > 0 else 0.0
            d = series[0] if series[0] > 0 else 0.0
            sse = 0

            for i in range(1, n):
                forecast = p * d
                sse += (series[i] - forecast) ** 2
                if series[i] > 0:
                    p = p + alpha * (1 - p)
                    d = d + beta * (series[i] - d)
                else:
                    p = p - alpha * p

            return sse

        try:
            result = minimize(objective, x0=[0.1, 0.1], bounds=[(0.01, 0.5), (0.01, 0.5)], method='L-BFGS-B')
            best_alpha, best_beta = result.x
        except Exception as exc:
            logger.debug("TSB optimization failed: %s", exc)
            best_alpha, best_beta = 0.1, 0.1

        p = 1.0 if series[0] > 0 else 0.0
        d = series[0] if series[0] > 0 else 0.0

        for i in range(1, n):
            if series[i] > 0:
                p = p + best_alpha * (1 - p)
                d = d + best_beta * (series[i] - d)
            else:
                p = p - best_alpha * p

        total_demand = p * d * horizon

        return max(0, total_demand)

    @staticmethod
    def median_forecast(series, horizon=5, window=5):
        series = np.array(series, dtype=float)
        if len(series) < window:
            median_val = np.median(series)
        else:
            median_val = np.median(series[-window:])
        return median_val * horizon

    @staticmethod
    def interval_based_forecast(series, horizon=5):
        series = np.array(series, dtype=float)
        if len(series) == 0:
            return 0

        non_zero_indices = np.where(series > 0)[0]
        if len(non_zero_indices) == 0:
            return 0

        non_zero_values = series[non_zero_indices]

        if len(non_zero_indices) > 1:
            intervals = np.diff(non_zero_indices)
            mean_interval = np.mean(intervals)
        else:
            mean_interval = len(series)

        mean_demand = np.mean(non_zero_values[-5:]) if len(non_zero_values) >= 5 else np.mean(non_zero_values)

        last_demand_idx = non_zero_indices[-1]
        periods_since_last = len(series) - last_demand_idx - 1

        if mean_interval > 0:
            expected_demands = horizon / mean_interval

            if periods_since_last >= mean_interval:
                expected_demands += 0.5
            elif periods_since_last > 0:
                adjustment = (periods_since_last / mean_interval) * 0.3
                expected_demands += adjustment
        else:
            expected_demands = horizon

        total_demand = expected_demands * mean_demand

        return max(0, total_demand)

    @staticmethod
    def croston(series, horizon=5, alpha=0.1):
        series = np.array(series, dtype=float)
        if len(series) == 0:
            return 0

        non_zero_indices = np.where(series > 0)[0]
        if len(non_zero_indices) == 0:
            return 0

        non_zero_values = series[non_zero_indices]

        if len(non_zero_indices) > 1:
            intervals = np.diff(non_zero_indices)
        else:
            intervals = np.array([len(series)])

        z = non_zero_values[0]
        p = intervals[0] if len(intervals) > 0 else len(series)

        for i in range(1, len(non_zero_values)):
            z = alpha * non_zero_values[i] + (1 - alpha) * z
            if i - 1 < len(intervals):
                p = alpha * intervals[i - 1] + (1 - alpha) * p

        if p > 0:
            forecast_per_period = z / p
        else:
            forecast_per_period = z / len(series)

        return max(0, forecast_per_period * horizon)

    @staticmethod
    def adida(series, horizon=5):
        series = np.array(series, dtype=float)
        n = len(series)
        if n == 0:
            return 0

        non_zero_count = np.sum(series > 0)
        if non_zero_count == 0:
            return 0

        avg_interval = n / non_zero_count if non_zero_count > 0 else n

        agg_level = max(1, int(np.ceil(avg_interval)))
        if agg_level > n // 2:
            agg_level = max(1, n // 4)

        n_agg = n // agg_level
        if n_agg < 2:
            return UltimateForecastMethods.interval_based_forecast(series, horizon)

        agg_series = []
        for i in range(n_agg):
            start = i * agg_level
            end = min(start + agg_level, n)
            agg_series.append(np.sum(series[start:end]))

        agg_series = np.array(agg_series, dtype=float)

        agg_forecast = np.mean(agg_series)

        per_period_forecast = agg_forecast / agg_level

        return max(0, per_period_forecast * horizon)

    @staticmethod
    def optimized_holt_winters(series, horizon=5, season_length=12):
        series = np.array(series, dtype=float)
        n = len(series)

        if n < season_length * 2:
            return UltimateForecastMethods.optimized_des(series, horizon)

        def objective(params):
            alpha, beta, gamma = params
            if alpha <= 0.01 or alpha >= 0.99 or beta <= 0 or beta >= 0.5 or gamma <= 0 or gamma >= 0.5:
                return 1e10

            level = np.mean(series[:season_length])
            trend = (np.mean(series[season_length:2 * season_length]) - level) / season_length
            seasonal = np.zeros(season_length)
            for i in range(season_length):
                seasonal[i] = series[i] - level

            sse = 0
            for i in range(season_length, n):
                forecast = level + trend + seasonal[i % season_length]
                sse += (series[i] - forecast) ** 2
                new_level = alpha * (series[i] - seasonal[i % season_length]) + (1 - alpha) * (level + trend)
                trend = beta * (new_level - level) + (1 - beta) * trend
                seasonal[i % season_length] = gamma * (series[i] - new_level) + (1 - gamma) * seasonal[i % season_length]
                level = new_level

            return sse

        try:
            result = minimize(
                objective,
                x0=[0.3, 0.1, 0.1],
                bounds=[(0.01, 0.99), (0.001, 0.3), (0.001, 0.3)],
                method='L-BFGS-B'
            )
            best_alpha, best_beta, best_gamma = result.x
        except Exception as exc:
            logger.debug("Holt-Winters optimization failed: %s", exc)
            best_alpha, best_beta, best_gamma = 0.3, 0.1, 0.1

        level = np.mean(series[:season_length])
        trend = (np.mean(series[season_length:2 * season_length]) - level) / season_length
        seasonal = np.zeros(season_length)
        for i in range(season_length):
            seasonal[i] = series[i] - level

        for i in range(season_length, n):
            new_level = best_alpha * (series[i] - seasonal[i % season_length]) + (1 - best_alpha) * (level + trend)
            trend = best_beta * (new_level - level) + (1 - best_beta) * trend
            seasonal[i % season_length] = best_gamma * (series[i] - new_level) + (1 - best_gamma) * seasonal[i % season_length]
            level = new_level

        total_forecast = 0
        for h in range(1, horizon + 1):
            forecast = level + h * trend + seasonal[(n + h) % season_length]
            total_forecast += max(0, forecast)

        return total_forecast

    @staticmethod
    def damped_trend(series, horizon=5):
        series = np.array(series, dtype=float)
        if len(series) < 4:
            return UltimateForecastMethods.optimized_ses(series, horizon)

        def objective(params):
            alpha, beta, phi = params
            if alpha <= 0.01 or alpha >= 0.99 or beta <= 0 or beta >= 0.5 or phi <= 0.8 or phi >= 1.0:
                return 1e10

            level = series[0]
            trend = series[1] - series[0]
            sse = 0

            for i in range(1, len(series)):
                forecast = level + phi * trend
                sse += (series[i] - forecast) ** 2
                new_level = alpha * series[i] + (1 - alpha) * (level + phi * trend)
                trend = beta * (new_level - level) + (1 - beta) * phi * trend
                level = new_level

            return sse

        try:
            result = minimize(
                objective,
                x0=[0.3, 0.1, 0.98],
                bounds=[(0.01, 0.99), (0.001, 0.3), (0.8, 1.0)],
                method='L-BFGS-B'
            )
            best_alpha, best_beta, best_phi = result.x
        except Exception as exc:
            logger.debug("Damped-trend optimization failed: %s", exc)
            best_alpha, best_beta, best_phi = 0.3, 0.1, 0.98

        level = series[0]
        trend = series[1] - series[0]

        for i in range(1, len(series)):
            new_level = best_alpha * series[i] + (1 - best_alpha) * (level + best_phi * trend)
            trend = best_beta * (new_level - level) + (1 - best_beta) * best_phi * trend
            level = new_level

        total_forecast = 0
        phi_sum = 0
        for h in range(1, horizon + 1):
            phi_sum += best_phi ** h
            total_forecast += max(0, level + phi_sum * trend)

        return total_forecast

    @staticmethod
    def theta_method(series, horizon=5):
        """Standard Theta method (Assimakopoulos & Nikolopoulos 2000).
        - theta=2 line: SES forecast
        - theta=0 line: linear trend extrapolation
        - equal-weight average of both lines' per-period forecasts.
        """
        series = np.array(series, dtype=float)
        n = len(series)
        if n < 3:
            return UltimateForecastMethods._mean_or_zero(series) * horizon

        x = np.arange(n, dtype=float)

        try:
            slope, intercept, _, _, _ = stats.linregress(x, series)
        except Exception as exc:
            logger.debug("Theta trend regression failed: %s", exc)
            return UltimateForecastMethods._mean_or_zero(series) * horizon

        # theta=2 line: 2 * series - (a + b*t) — dilates series around trend
        trend_at_x = intercept + slope * x
        theta2_line = 2.0 * series - trend_at_x

        # SES level for theta=2 line (per-period forecast)
        ses_level = UltimateForecastMethods.optimized_ses(theta2_line, horizon=1)

        # theta=0 line is the linear trend itself; extrapolate per-period
        trend_per_period = intercept + slope * (n + (horizon + 1) / 2.0)

        # equal-weight average
        per_period = 0.5 * ses_level + 0.5 * trend_per_period

        return max(0, per_period * horizon)

    @staticmethod
    def weighted_moving_average(series, horizon=5, window=5):
        series = np.array(series, dtype=float)
        if len(series) < window:
            weights = np.arange(1, len(series) + 1, dtype=float)
            wma = np.average(series, weights=weights)
        else:
            recent = series[-window:]
            weights = np.arange(1, window + 1, dtype=float)
            wma = np.average(recent, weights=weights)

        return wma * horizon

    @staticmethod
    def winsorized_mean(series, horizon=5, window=8, lower_pct=10, upper_pct=90):
        series = np.array(series, dtype=float)
        if len(series) < 3:
            return UltimateForecastMethods._mean_or_zero(series) * horizon

        if len(series) >= window:
            data = series[-window:]
        else:
            data = series

        lower_bound = np.percentile(data, lower_pct)
        upper_bound = np.percentile(data, upper_pct)

        winsorized = np.clip(data, lower_bound, upper_bound)

        return np.mean(winsorized) * horizon

    @staticmethod
    def naive_with_drift(series, horizon=5):
        series = np.array(series, dtype=float)
        n = len(series)
        if n < 2:
            return series[-1] * horizon if n == 1 else 0

        drift = (series[-1] - series[0]) / (n - 1)

        total_forecast = 0
        for h in range(1, horizon + 1):
            total_forecast += max(0, series[-1] + h * drift)

        return total_forecast


class AdaptiveModelSelector:

    @staticmethod
    def select_best_method(series, test_size=5, disabled_methods=None):
        series = np.array(series, dtype=float)
        disabled_methods = set(disabled_methods or [])

        if np.sum(series) == 0:
            return 'zero', {
                'method': 'zero',
                'mase': 0,
                'total_error': 0,
                'forecast_total': 0,
                'actual_total': 0,
                'pattern': 'all_zero',
                'analysis': {'zero_ratio': 1.0, 'cv': 0.0},
                'drift_info': None,
            }, {}

        if len(series) <= test_size:
            forecast_total = float(np.sum(series))
            actual_total = float(np.sum(series))
            return 'moving_average', {
                'method': 'moving_average',
                'mase': 0,
                'total_error': 0,
                'forecast_total': forecast_total,
                'actual_total': actual_total,
                'pattern': 'unknown',
                'analysis': {'zero_ratio': float(np.mean(series == 0)) if len(series) else 1.0, 'cv': 0.0},
                'drift_info': None,
            }, {}

        train = series[:-test_size]
        test = series[-test_size:]
        actual_total = np.sum(test)

        adaptive_analyzer = AdaptiveDataAnalyzer(train)
        pattern = adaptive_analyzer.pattern
        analysis = adaptive_analyzer.analysis
        drift_info = adaptive_analyzer.drift_info

        methods_to_test = AdaptiveModelSelector._get_candidate_methods(pattern, analysis, test_size)
        methods_to_test = {
            method_name: forecast_func
            for method_name, forecast_func in methods_to_test.items()
            if method_name not in disabled_methods
        }

        methods_results = {}

        for method_name, forecast_func in methods_to_test.items():
            try:
                forecast_total = forecast_func(train)
                mase, total_error = AdaptiveModelSelector._calculate_metrics(forecast_total, actual_total, train, test_size)
                methods_results[method_name] = {
                    'mase': mase,
                    'total_error': total_error,
                    'forecast_total': forecast_total,
                    'error': None,
                }
            except Exception as exc:
                logger.warning("Forecast method %s failed: %s", method_name, exc)
                methods_results[method_name] = {
                    'mase': 1e10,
                    'total_error': abs(actual_total),
                    'forecast_total': 0,
                    'error': str(exc),
                }

        valid_results = {k: v for k, v in methods_results.items() if v['mase'] < 1e10}

        if valid_results:
            best_method = min(valid_results.items(), key=lambda x: x[1]['mase'])
        else:
            best_method = min(methods_results.items(), key=lambda x: x[1]['mase'])

        return best_method[0], {
            'method': best_method[0],
            'mase': best_method[1]['mase'],
            'total_error': best_method[1]['total_error'],
            'forecast_total': best_method[1]['forecast_total'],
            'actual_total': actual_total,
            'pattern': pattern,
            'analysis': analysis,
            'drift_info': drift_info
        }, methods_results

    @staticmethod
    def _get_candidate_methods(pattern, analysis, test_size):
        FM = UltimateForecastMethods

        core_methods = {}
        extended_methods = {}
        universal_methods = {
            'ma_3': lambda s: FM.moving_average(s, horizon=test_size, window=3),
            'ma_5': lambda s: FM.moving_average(s, horizon=test_size, window=5),
            'weighted_ma_5': lambda s: FM.weighted_moving_average(s, horizon=test_size, window=5),
        }

        if pattern == 'seasonal':
            core_methods = {
                'optimized_hw': lambda s: FM.optimized_holt_winters(s, horizon=test_size),
                'seasonal_naive': lambda s: FM.seasonal_naive(s, horizon=test_size),
                'optimized_des': lambda s: FM.optimized_des(s, horizon=test_size),
                'theta': lambda s: FM.theta_method(s, horizon=test_size),
            }
            extended_methods = {
                'optimized_ses': lambda s: FM.optimized_ses(s, horizon=test_size),
                'median_5': lambda s: FM.median_forecast(s, horizon=test_size, window=5),
            }

        elif pattern == 'trending':
            core_methods = {
                'optimized_des': lambda s: FM.optimized_des(s, horizon=test_size),
                'optimized_hw': lambda s: FM.optimized_holt_winters(s, horizon=test_size),
            }
            extended_methods = {
                'theta': lambda s: FM.theta_method(s, horizon=test_size),
                'optimized_ses': lambda s: FM.optimized_ses(s, horizon=test_size),
                'seasonal_naive': lambda s: FM.seasonal_naive(s, horizon=test_size),
            }

        elif pattern == 'sparse':
            core_methods = {
                'interval_based': lambda s: FM.interval_based_forecast(s, horizon=test_size),
                'median_5': lambda s: FM.median_forecast(s, horizon=test_size, window=5),
                'croston': lambda s: FM.croston(s, horizon=test_size),
                'adida': lambda s: FM.adida(s, horizon=test_size),
            }
            extended_methods = {
                'tsb_opt': lambda s: FM.tsb_opt(s, horizon=test_size),
                'winsorized': lambda s: FM.winsorized_mean(s, horizon=test_size),
            }

        elif pattern == 'lumpy':
            core_methods = {
                'tsb_opt': lambda s: FM.tsb_opt(s, horizon=test_size),
                'croston': lambda s: FM.croston(s, horizon=test_size),
                'adida': lambda s: FM.adida(s, horizon=test_size),
                'interval_based': lambda s: FM.interval_based_forecast(s, horizon=test_size),
            }
            extended_methods = {
                'winsorized': lambda s: FM.winsorized_mean(s, horizon=test_size),
                'median_5': lambda s: FM.median_forecast(s, horizon=test_size, window=5),
                'theta': lambda s: FM.theta_method(s, horizon=test_size),
            }

        elif pattern == 'intermittent':
            core_methods = {
                'tsb_opt': lambda s: FM.tsb_opt(s, horizon=test_size),
                'croston': lambda s: FM.croston(s, horizon=test_size),
                'adida': lambda s: FM.adida(s, horizon=test_size),
                'interval_based': lambda s: FM.interval_based_forecast(s, horizon=test_size),
                'median_5': lambda s: FM.median_forecast(s, horizon=test_size, window=5),
            }
            extended_methods = {
                'optimized_ses': lambda s: FM.optimized_ses(s, horizon=test_size),
                'winsorized': lambda s: FM.winsorized_mean(s, horizon=test_size),
                'theta': lambda s: FM.theta_method(s, horizon=test_size),
            }

        elif pattern == 'erratic':
            core_methods = {
                'winsorized': lambda s: FM.winsorized_mean(s, horizon=test_size),
                'median_5': lambda s: FM.median_forecast(s, horizon=test_size, window=5),
                'adida': lambda s: FM.adida(s, horizon=test_size),
                'interval_based': lambda s: FM.interval_based_forecast(s, horizon=test_size),
            }
            extended_methods = {
                'tsb_opt': lambda s: FM.tsb_opt(s, horizon=test_size),
                'croston': lambda s: FM.croston(s, horizon=test_size),
                'theta': lambda s: FM.theta_method(s, horizon=test_size),
                'optimized_ses': lambda s: FM.optimized_ses(s, horizon=test_size),
            }

        elif pattern == 'stable':
            core_methods = {
                'optimized_ses': lambda s: FM.optimized_ses(s, horizon=test_size),
                'ma_3': lambda s: FM.moving_average(s, horizon=test_size, window=3),
                'ma_5': lambda s: FM.moving_average(s, horizon=test_size, window=5),
                'theta': lambda s: FM.theta_method(s, horizon=test_size),
                'interval_based': lambda s: FM.interval_based_forecast(s, horizon=test_size),
            }
            extended_methods = {
                'weighted_ma_5': lambda s: FM.weighted_moving_average(s, horizon=test_size, window=5),
                'optimized_des': lambda s: FM.optimized_des(s, horizon=test_size),
                'optimized_hw': lambda s: FM.optimized_holt_winters(s, horizon=test_size),
                'seasonal_naive': lambda s: FM.seasonal_naive(s, horizon=test_size),
            }

        else:
            core_methods = {
                'ma_5': lambda s: FM.moving_average(s, horizon=test_size, window=5),
                'optimized_ses': lambda s: FM.optimized_ses(s, horizon=test_size),
                'median_5': lambda s: FM.median_forecast(s, horizon=test_size, window=5),
            }
            extended_methods = {}

        all_methods = {}
        all_methods.update(core_methods)
        all_methods.update(extended_methods)
        all_methods.update(universal_methods)

        seen = set()
        unique_methods = {}
        for k, v in all_methods.items():
            if k not in seen:
                seen.add(k)
                unique_methods[k] = v

        return unique_methods

    @staticmethod
    def _calculate_metrics(forecast_total, actual_total, train=None, horizon=5):
        total_error = abs(forecast_total - actual_total)

        mase = 0.0
        if train is not None and len(train) > 1:
            avg_forecast = forecast_total / horizon if horizon > 0 else 0
            avg_actual = actual_total / horizon if horizon > 0 else 0

            mae = abs(avg_forecast - avg_actual)

            naive_errors = np.abs(np.diff(train))

            if len(naive_errors) > 0 and np.mean(naive_errors) > 1e-10:
                mase = mae / np.mean(naive_errors)
            else:
                mase = mae if mae > 0 else 0.0

        return mase, total_error


logger = logging.getLogger(__name__)


def _first_existing_path(candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    checked = ", ".join(str(Path(candidate)) for candidate in candidates)
    raise FileNotFoundError(f"未找到输入文件，已检查: {checked}")


def _is_v6_new_method(method):
    return "是" if method in NEW_V6_METHODS else "否"


def _method_family(method):
    if method == "zero":
        return "特殊方法"
    if method in NEW_V6_METHODS:
        return "V6新增"
    if method in LEGACY_METHODS:
        return "原有方法"
    return "其他"


def _baseline_metrics(train, actual_total, horizon):
    train = np.array(train, dtype=float)
    if len(train) == 0 or horizon <= 0:
        baseline_forecast = 0.0
    elif len(train) >= 5:
        baseline_forecast = float(np.mean(train[-5:]) * horizon)
    else:
        baseline_forecast = float(np.mean(train) * horizon)

    baseline_error = abs(baseline_forecast - actual_total)
    baseline_mae = abs(baseline_forecast / horizon - actual_total / horizon) if horizon > 0 else 0.0
    naive_errors = np.abs(np.diff(train))
    mean_naive_error = np.mean(naive_errors) if len(naive_errors) > 0 else 0.0
    if mean_naive_error > 1e-10:
        baseline_mase = baseline_mae / mean_naive_error
    else:
        baseline_mase = baseline_mae if baseline_mae > 0 else 0.0

    return baseline_mase, baseline_error


def _mean(values):
    return float(np.mean(values)) if values else 0.0


class AdaptiveForecaster:

    def __init__(self, data_path, test_periods=5, disabled_methods=None):
        self.data_path = Path(data_path)
        self.data = pd.read_csv(self.data_path)
        self.test_periods = test_periods
        self.disabled_methods = set(disabled_methods or [])
        self.materials = self.data.iloc[:, 0].values
        self.time_series = self.data.iloc[:, 1:].values
        # Pre-build material-to-index lookup to avoid O(n²) .index() calls
        self._material_index = {str(m): i for i, m in enumerate(self.materials)}
        self.results = {}
        self.summary = {}
        self.drift_analysis = {}
        self.all_zero_materials = []

    def _series_for(self, material):
        return self.time_series[self._material_index[str(material)]]

    def run_analysis(self):
        print("=" * 80)
        print("智能物料需求预测系统 - 扩展方法库版 (v6.1)")
        print("改进：方法库从9种扩展到17种 + all_zero物料纳入整体评估")
        print("=" * 80)
        print(f"\n数据概览:")
        print(f"  - 物料数量: {len(self.materials)}")
        print(f"  - 历史周期数: {self.time_series.shape[1]}")
        print(f"  - 测试集周期数: {self.test_periods}")
        print(f"  - 训练集周期数: {self.time_series.shape[1] - self.test_periods}")
        print(f"  - 预测方式: 一次性预测{self.test_periods}期总需求")
        print(f"  - 最优区间范围: {OptimalIntervalFinder.MIN_INTERVAL}-{OptimalIntervalFinder.MAX_INTERVAL}周")
        print(f"\n方法库:")
        print(f"  - 主方法库: ma_3, ma_5, optimized_ses, optimized_des,")
        print(f"    seasonal_naive, tsb_opt, median_5, interval_based, croston,")
        print(f"    adida, optimized_hw, theta, weighted_ma_5, winsorized")
        print(f"  - 暂移出主方法库: sba, naive_drift, damped_trend")
        print(f"  - 已删除方法: ma_7, holt_winters")
        if self.disabled_methods:
            print(f"  - 本次实验禁用方法: {', '.join(sorted(self.disabled_methods))}")

        print("\n正在分析各物料数据特点并自动寻优最新区间...")
        print("主要评估指标: MASE (Mean Absolute Scaled Error)")
        print("分类策略: 自适应模式分类 + 扩展方法库 + 跨模式方法测试")

        for idx, material in enumerate(self.materials):
            if (idx + 1) % RuntimeConfig.progress_every == 0:
                print(f"  已处理: {idx + 1}/{len(self.materials)} 个物料")

            series = self.time_series[idx]

            if np.sum(series) == 0:
                self.results[material] = {
                    'best_method': 'zero',
                    'forecast_total': 0,
                    'actual_total': 0,
                    'mase': 0,
                    'total_error': 0,
                    'pattern': 'all_zero',
                    'analysis': {'zero_ratio': 1.0, 'cv': 0.0},
                    'optimal_interval': 0,
                    'drift_info': None,
                    'all_methods_results': {}
                }
                self.all_zero_materials.append(material)
                continue

            best_method_name, method_info, all_results = AdaptiveModelSelector.select_best_method(
                series, self.test_periods, disabled_methods=self.disabled_methods)

            self.results[material] = {
                'best_method': best_method_name,
                'forecast_total': method_info['forecast_total'],
                'actual_total': method_info['actual_total'],
                'mase': method_info['mase'],
                'total_error': method_info['total_error'],
                'pattern': method_info['pattern'],
                'analysis': method_info['analysis'],
                'optimal_interval': method_info['drift_info']['optimal_interval'] if method_info['drift_info'] else 0,
                'drift_info': method_info['drift_info'],
                'all_methods_results': all_results
            }

        print(f"  已处理: {len(self.materials)}/{len(self.materials)} 个物料")
        print(f"  全历史需求为0的物料: {len(self.all_zero_materials)} 个（已纳入平均指标计算）")
        self._generate_summary()
        self._analyze_drift()

    def _generate_summary(self):
        method_counts = {}
        pattern_counts = {}
        total_mase = []
        total_error_list = []
        total_intervals = []

        baseline_mase = []
        baseline_total_error = []

        v6_new_method_counts = {}
        v6_new_method_mase = []

        for material, result in self.results.items():
            method = result['best_method']
            pattern = result['pattern']

            method_counts[method] = method_counts.get(method, 0) + 1
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            if result['mase'] is not None and not np.isinf(result['mase']):
                total_mase.append(result['mase'])
                total_error_list.append(result['total_error'])

            if result['optimal_interval'] > 0:
                total_intervals.append(result['optimal_interval'])

            if method in NEW_V6_METHODS:
                v6_new_method_counts[method] = v6_new_method_counts.get(method, 0) + 1
                if result['mase'] is not None and not np.isinf(result['mase']):
                    v6_new_method_mase.append(result['mase'])

            series = self._series_for(material)
            train = series[:-self.test_periods]
            test = series[-self.test_periods:]
            actual_total = np.sum(test)

            baseline_mase_val, baseline_error_val = _baseline_metrics(train, actual_total, self.test_periods)
            baseline_mase.append(baseline_mase_val)
            baseline_total_error.append(baseline_error_val)

        final_all_zero_count = pattern_counts.get('all_zero', 0)
        included_materials_count = len(self.materials)
        non_all_zero_count = len(self.materials) - final_all_zero_count

        avg_mase = _mean(total_mase)
        avg_total_error = _mean(total_error_list)
        baseline_avg_mase = _mean(baseline_mase)
        baseline_avg_total_error = _mean(baseline_total_error)

        self.summary = {
            'method_distribution': method_counts,
            'pattern_distribution': pattern_counts,
            'avg_mase': avg_mase,
            'avg_total_error': avg_total_error,
            'avg_optimal_interval': _mean(total_intervals),
            'baseline_avg_mase': baseline_avg_mase,
            'baseline_avg_total_error': baseline_avg_total_error,
            'improvement_mase': (baseline_avg_mase - avg_mase) / baseline_avg_mase * 100 if baseline_avg_mase > 0 else 0,
            'improvement_total_error': (baseline_avg_total_error - avg_total_error) / baseline_avg_total_error * 100 if baseline_avg_total_error > 0 else 0,
            'v6_new_method_counts': v6_new_method_counts,
            'v6_new_method_avg_mase': _mean(v6_new_method_mase),
            'v6_new_method_total_uses': sum(v6_new_method_counts.values()),
            'all_zero_count': final_all_zero_count,
            'full_history_all_zero_count': len(self.all_zero_materials),
            'included_materials_count': included_materials_count,
            'non_all_zero_count': non_all_zero_count,
            'valid_materials_count': included_materials_count,
        }

    def _analyze_drift(self):
        drift_count = 0
        pattern_transitions = defaultdict(int)
        drift_examples = []

        for material, result in self.results.items():
            drift_info = result.get('drift_info')
            if drift_info is None:
                continue

            if drift_info['pattern_shift']:
                drift_count += 1
                transition = f"{drift_info['full_pattern']}→{drift_info['optimal_pattern']}"
                pattern_transitions[transition] += 1

                if len(drift_examples) < 20:
                    drift_examples.append({
                        'material': material,
                        'full_pattern': drift_info['full_pattern'],
                        'optimal_pattern': drift_info['optimal_pattern'],
                        'optimal_interval': drift_info['optimal_interval'],
                        'full_zero_ratio': drift_info['full_zero_ratio'],
                        'optimal_zero_ratio': drift_info['optimal_zero_ratio'],
                        'full_cv': drift_info['full_cv'],
                        'optimal_cv': drift_info['optimal_cv'],
                        'final_pattern': result['pattern'],
                    })

        self.drift_analysis = {
            'total_materials': len(self.materials),
            'drift_count': drift_count,
            'drift_ratio': drift_count / len(self.materials) * 100 if len(self.materials) > 0 else 0,
            'pattern_transitions': dict(pattern_transitions),
            'drift_examples': drift_examples,
        }

    def print_summary(self):
        print("\n" + "=" * 80)
        print("预测结果汇总 (扩展方法库版 - v6.1)")
        print("=" * 80)

        print(f"\n【物料统计】")
        print(f"  总物料数: {len(self.materials)}")
        print(f"  all_zero物料: {self.summary['all_zero_count']} 个（按最终分类，已纳入平均指标计算）")
        print(f"  全历史需求为0物料: {self.summary['full_history_all_zero_count']} 个")
        print(f"  非all_zero物料: {self.summary['non_all_zero_count']} 个")
        print(f"  纳入指标计算物料数: {self.summary['included_materials_count']} 个")

        print("\n【三层七模式分布（自适应分类）】")
        print("  第一层 - 宏观结构特征:")
        for pattern in ['seasonal', 'trending']:
            count = self.summary['pattern_distribution'].get(pattern, 0)
            pct = count / len(self.materials) * 100
            print(f"    {pattern:15s}: {count:4d} 个物料 ({pct:5.1f}%)")

        print("\n  第二层 - 极端长尾特征:")
        count = self.summary['pattern_distribution'].get('sparse', 0)
        pct = count / len(self.materials) * 100
        print(f"    {'sparse':15s}: {count:4d} 个物料 ({pct:5.1f}%)")
        count = self.summary['pattern_distribution'].get('all_zero', 0)
        pct = count / len(self.materials) * 100
        print(f"    {'all_zero':15s}: {count:4d} 个物料 ({pct:5.1f}%)")

        print("\n  第三层 - SBC矩阵四象限:")
        for pattern in ['lumpy', 'intermittent', 'erratic', 'stable']:
            count = self.summary['pattern_distribution'].get(pattern, 0)
            pct = count / len(self.materials) * 100
            print(f"    {pattern:15s}: {count:4d} 个物料 ({pct:5.1f}%)")

        print("\n【最优预测方法分布】")
        sorted_methods = sorted(self.summary['method_distribution'].items(), key=lambda x: -x[1])
        for method, count in sorted_methods:
            pct = count / len(self.materials) * 100
            tag = " [V6新增]" if method in NEW_V6_METHODS else ""
            print(f"  {method:25s}: {count:4d} 个物料 ({pct:5.1f}%){tag}")

        print("\n【V6新增方法使用统计】")
        v6_new = self.summary.get('v6_new_method_counts', {})
        if v6_new:
            total_new_uses = self.summary.get('v6_new_method_total_uses', 0)
            print(f"  V6新增方法被采用总数: {total_new_uses} 个物料 ({total_new_uses / len(self.materials) * 100:.1f}%)")
            for method, count in sorted(v6_new.items(), key=lambda x: -x[1]):
                pct = count / len(self.materials) * 100
                print(f"    {method:25s}: {count:4d} 个物料 ({pct:5.1f}%)")
            if self.summary.get('v6_new_method_avg_mase', 0) > 0:
                print(f"  V6新增方法平均MASE: {self.summary['v6_new_method_avg_mase']:.4f}")
        else:
            print("  无V6新增方法被采用")

        print("\n【最优时间区间分析】")
        print(f"  平均最优区间长度: {self.summary['avg_optimal_interval']:.1f} 周")
        print(f"  >>>>>>>>>> 区间范围: {OptimalIntervalFinder.MIN_INTERVAL}-{OptimalIntervalFinder.MAX_INTERVAL} 周")

        print("\n【预测性能对比】（含all_zero物料）")
        print(f"  基准方法 (五期移动平均):")
        print(f"    - 平均MASE: {self.summary['baseline_avg_mase']:.4f}")
        print(f"    - 平均总量误差: {self.summary['baseline_avg_total_error']:.4f}")
        print(f"\n  V6.1扩展方法库方法:")
        print(f"    - 平均MASE: {self.summary['avg_mase']:.4f}")
        print(f"    - 平均总量误差: {self.summary['avg_total_error']:.4f}")

        print(f"\n【性能提升】")
        print(f"  MASE提升: {self.summary['improvement_mase']:.2f}%")
        print(f"  总量误差提升: {self.summary['improvement_total_error']:.2f}%")

        self._print_drift_analysis()

    def _print_drift_analysis(self):
        da = self.drift_analysis

        print("\n" + "=" * 80)
        print("概念漂移分析报告")
        print("=" * 80)

        print(f"\n【漂移检测概览】")
        print(f"  总物料数: {da['total_materials']}")
        print(f"  检测到模式漂移的物料: {da['drift_count']} ({da['drift_ratio']:.1f}%)")

        if da['pattern_transitions']:
            print(f"\n【模式转换类型分布】")
            sorted_transitions = sorted(da['pattern_transitions'].items(), key=lambda x: -x[1])
            for transition, count in sorted_transitions:
                pct = count / da['drift_count'] * 100 if da['drift_count'] > 0 else 0
                print(f"  {transition:30s}: {count:4d} 个物料 ({pct:5.1f}%)")

        if da['drift_examples']:
            print(f"\n【模式漂移示例（前10个）】")
            print(f"  {'通用码':<12s} {'全历史':>10s} {'最优':>10s} {'最终':>10s} {'区间':>6s} {'零值比变化':>12s}")
            print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*12}")
            for ex in da['drift_examples'][:10]:
                zr_change = f"{ex['full_zero_ratio']:.2f}→{ex['optimal_zero_ratio']:.2f}"
                mat_str = str(ex['material'])[:12]
                print(f"  {mat_str:<12s} {ex['full_pattern']:>10s} {ex['optimal_pattern']:>10s} "
                      f"{ex['final_pattern']:>10s} {ex['optimal_interval']:>6d} {zr_change:>12s}")

    def save_results(self, output_path):
        rows = []
        for material in self.materials:
            result = self.results[material]
            drift_info = result.get('drift_info', {})

            series = self._series_for(material)
            train = series[:-self.test_periods]
            test = series[-self.test_periods:]
            actual_total = np.sum(test)

            baseline_mase, baseline_error = _baseline_metrics(train, actual_total, self.test_periods)
            smart_mase = result['mase']
            smart_error = result['total_error']

            mase_improve = (baseline_mase - smart_mase) / baseline_mase * 100 if baseline_mase > 0 else 0
            error_improve = (baseline_error - smart_error) / baseline_error * 100 if baseline_error > 0 else 0

            row = {
                '通用码': material,
                '数据模式(自适应)': result['pattern'],
                '数据模式(全历史)': drift_info.get('full_pattern', result['pattern']) if drift_info else result['pattern'],
                '数据模式(最优区间)': drift_info.get('optimal_pattern', result['pattern']) if drift_info else result['pattern'],
                '模式漂移': '是' if (drift_info and drift_info.get('pattern_shift', False)) else '否',
                '最优区间(周)': drift_info.get('optimal_interval', 0) if drift_info else 0,
                '最优方法': result['best_method'],
                '是否V6新增方法': _is_v6_new_method(result['best_method']),
                '预测总量': round(result['forecast_total'], 2),
                '实际总量': round(result['actual_total'], 2),
                '绝对总量误差': round(result['total_error'], 2),
                'MASE': round(result['mase'], 4),
                '移动平均MASE': round(baseline_mase, 4),
                'MASE提升(%)': round(mase_improve, 2),
                '移动平均总量误差': round(baseline_error, 2),
                '总量误差提升(%)': round(error_improve, 2),
                '零值比例(全历史)': round(drift_info.get('full_zero_ratio', result['analysis']['zero_ratio']), 4) if drift_info else round(result['analysis']['zero_ratio'], 4),
                '零值比例(最优)': round(drift_info.get('optimal_zero_ratio', result['analysis']['zero_ratio']), 4) if drift_info else round(result['analysis']['zero_ratio'], 4),
                '变异系数(全历史)': round(drift_info.get('full_cv', result['analysis']['cv']), 4) if drift_info else round(result['analysis']['cv'], 4),
                '变异系数(最优)': round(drift_info.get('optimal_cv', result['analysis']['cv']), 4) if drift_info else round(result['analysis']['cv'], 4),
            }
            rows.append(row)

        df_results = pd.DataFrame(rows)
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已保存至: {output_path}")

    def save_comparison_report(self, output_path):
        comparison_rows = []

        for material in self.materials:
            result = self.results[material]

            series = self._series_for(material)
            train = series[:-self.test_periods]
            test = series[-self.test_periods:]
            actual_total = np.sum(test)

            baseline_mase, baseline_error = _baseline_metrics(train, actual_total, self.test_periods)
            smart_mase = result['mase']
            smart_error = result['total_error']

            mase_improve = (baseline_mase - smart_mase) / baseline_mase * 100 if baseline_mase > 0 else 0
            error_improve = (baseline_error - smart_error) / baseline_error * 100 if baseline_error > 0 else 0

            comparison_rows.append({
                '通用码': material,
                '数据模式': result['pattern'],
                '最优方法': result['best_method'],
                '是否V6新增方法': _is_v6_new_method(result['best_method']),
                '最优区间(周)': result['optimal_interval'],
                '移动平均MASE': round(baseline_mase, 4),
                '智能方法MASE': round(smart_mase, 4),
                'MASE提升(%)': round(mase_improve, 2),
                '移动平均总量误差': round(baseline_error, 2),
                '智能方法总量误差': round(smart_error, 2),
                '总量误差提升(%)': round(error_improve, 2),
            })

        df_comparison = pd.DataFrame(comparison_rows)
        df_comparison.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"对比报告已保存至: {output_path}")

    def save_drift_report(self, output_path):
        da = self.drift_analysis

        rows = []
        for material in self.materials:
            result = self.results[material]
            drift_info = result.get('drift_info')
            if drift_info is None:
                continue

            rows.append({
                '通用码': material,
                '全历史模式': drift_info['full_pattern'],
                '最优区间模式': drift_info['optimal_pattern'],
                '最终采用模式': result['pattern'],
                '模式漂移': '是' if drift_info['pattern_shift'] else '否',
                '最优区间长度': drift_info['optimal_interval'],
                '零值比(全历史)': round(drift_info['full_zero_ratio'], 4),
                '零值比(最优区间)': round(drift_info['optimal_zero_ratio'], 4),
                'CV(全历史)': round(drift_info['full_cv'], 4),
                'CV(最优区间)': round(drift_info['optimal_cv'], 4),
                '趋势(全历史)': drift_info['full_trend'],
                '趋势(最优区间)': drift_info['optimal_trend'],
                'MASE': round(result['mase'], 4),
                '最优方法': result['best_method'],
                '是否V6新增方法': _is_v6_new_method(result['best_method']),
            })

        df_drift = pd.DataFrame(rows)
        df_drift.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"漂移分析报告已保存至: {output_path}")

    def save_method_contribution_report(self, output_path):
        stats_by_method = defaultdict(lambda: {
            'tested_count': 0,
            'winner_count': 0,
            'top3_count': 0,
            'mase_values': [],
            'total_error_values': [],
            'rank_values': [],
            'pattern_counts': defaultdict(int),
            'winner_pattern_counts': defaultdict(int),
        })

        for material in self.materials:
            result = self.results[material]
            pattern = result['pattern']
            best_method = result['best_method']
            all_results = result.get('all_methods_results', {})

            if best_method == 'zero' and not all_results:
                all_results = {
                    'zero': {
                        'mase': result['mase'],
                        'total_error': result['total_error'],
                        'forecast_total': result['forecast_total'],
                        'error': None,
                    }
                }

            valid_results = {
                method: metrics for method, metrics in all_results.items()
                if metrics.get('mase') is not None and np.isfinite(metrics.get('mase')) and metrics.get('mase') < 1e10
            }
            ranked_methods = sorted(valid_results.items(), key=lambda item: item[1]['mase'])
            ranks = {method: rank for rank, (method, _) in enumerate(ranked_methods, start=1)}

            for method, metrics in valid_results.items():
                item = stats_by_method[method]
                item['tested_count'] += 1
                item['mase_values'].append(metrics['mase'])
                item['total_error_values'].append(metrics['total_error'])
                item['rank_values'].append(ranks[method])
                item['pattern_counts'][pattern] += 1

                if ranks[method] <= 3:
                    item['top3_count'] += 1

            if best_method in stats_by_method:
                stats_by_method[best_method]['winner_count'] += 1
                stats_by_method[best_method]['winner_pattern_counts'][pattern] += 1

        rows = []
        for method, item in stats_by_method.items():
            tested_count = item['tested_count']
            winner_count = item['winner_count']
            rows.append({
                '方法': method,
                '方法类型': _method_family(method),
                '被测试次数': tested_count,
                '胜出次数': winner_count,
                '胜出率(%)': round(winner_count / tested_count * 100, 2) if tested_count else 0,
                'Top3次数': item['top3_count'],
                'Top3率(%)': round(item['top3_count'] / tested_count * 100, 2) if tested_count else 0,
                '平均MASE': round(_mean(item['mase_values']), 4),
                '中位MASE': round(float(np.median(item['mase_values'])), 4) if item['mase_values'] else 0,
                '平均总量误差': round(_mean(item['total_error_values']), 2),
                '平均排名': round(_mean(item['rank_values']), 2),
                '覆盖模式数': len(item['pattern_counts']),
                '被测试模式分布': '; '.join(f"{k}:{v}" for k, v in sorted(item['pattern_counts'].items())),
                '胜出模式分布': '; '.join(f"{k}:{v}" for k, v in sorted(item['winner_pattern_counts'].items())),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                by=['胜出次数', 'Top3率(%)', '平均排名', '平均MASE'],
                ascending=[False, False, True, True],
            )
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"方法贡献报告已保存至: {output_path}")

    def save_method_overlap_report(self, output_path):
        pair_stats = defaultdict(lambda: {
            'shared_count': 0,
            'first_better_count': 0,
            'second_better_count': 0,
            'tie_count': 0,
            'first_mase_values': [],
            'second_mase_values': [],
            'mase_diff_values': [],
            'first_forecasts': [],
            'second_forecasts': [],
        })

        for result in self.results.values():
            all_results = result.get('all_methods_results', {})
            if result['best_method'] == 'zero' and not all_results:
                all_results = {
                    'zero': {
                        'mase': result['mase'],
                        'total_error': result['total_error'],
                        'forecast_total': result['forecast_total'],
                        'error': None,
                    }
                }

            valid_results = {
                method: metrics for method, metrics in all_results.items()
                if metrics.get('mase') is not None and np.isfinite(metrics.get('mase')) and metrics.get('mase') < 1e10
            }
            methods = sorted(valid_results)

            for i, first in enumerate(methods):
                for second in methods[i + 1:]:
                    first_metrics = valid_results[first]
                    second_metrics = valid_results[second]
                    key = (first, second)
                    item = pair_stats[key]
                    first_mase = first_metrics['mase']
                    second_mase = second_metrics['mase']

                    item['shared_count'] += 1
                    item['first_mase_values'].append(first_mase)
                    item['second_mase_values'].append(second_mase)
                    item['mase_diff_values'].append(abs(first_mase - second_mase))
                    item['first_forecasts'].append(first_metrics.get('forecast_total', 0))
                    item['second_forecasts'].append(second_metrics.get('forecast_total', 0))

                    if first_mase < second_mase:
                        item['first_better_count'] += 1
                    elif second_mase < first_mase:
                        item['second_better_count'] += 1
                    else:
                        item['tie_count'] += 1

        rows = []
        for (first, second), item in pair_stats.items():
            shared_count = item['shared_count']
            first_better = item['first_better_count']
            second_better = item['second_better_count']
            tie_count = item['tie_count']
            forecasts_a = np.array(item['first_forecasts'], dtype=float)
            forecasts_b = np.array(item['second_forecasts'], dtype=float)
            if shared_count >= 2 and np.std(forecasts_a) > 1e-10 and np.std(forecasts_b) > 1e-10:
                corr = float(np.corrcoef(forecasts_a, forecasts_b)[0, 1])
            else:
                corr = 1.0 if np.allclose(forecasts_a, forecasts_b) else 0.0

            rows.append({
                '方法A': first,
                '方法B': second,
                '共同测试次数': shared_count,
                'A更优次数': first_better,
                'B更优次数': second_better,
                '持平次数': tie_count,
                'A更优率(%)': round(first_better / shared_count * 100, 2) if shared_count else 0,
                'B更优率(%)': round(second_better / shared_count * 100, 2) if shared_count else 0,
                '持平率(%)': round(tie_count / shared_count * 100, 2) if shared_count else 0,
                'A平均MASE': round(_mean(item['first_mase_values']), 4),
                'B平均MASE': round(_mean(item['second_mase_values']), 4),
                '平均MASE绝对差': round(_mean(item['mase_diff_values']), 4),
                '预测总量相关系数': round(corr, 4),
                '疑似重复': '是' if shared_count >= 20 and corr >= 0.98 and _mean(item['mase_diff_values']) <= 0.01 else '否',
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                by=['疑似重复', '预测总量相关系数', '平均MASE绝对差', '共同测试次数'],
                ascending=[False, False, True, False],
            )
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"方法重叠度报告已保存至: {output_path}")

    def save_method_family_report(self, output_path):
        method_groups = {
            '间歇需求方法组': {'croston', 'adida', 'tsb_opt'},
            '季节趋势方法组': {'optimized_hw', 'seasonal_naive', 'optimized_des'},
        }

        rows = []
        for group_name, group_methods in method_groups.items():
            stats_by_method = defaultdict(lambda: {
                'tested_count': 0,
                'best_count': 0,
                'top2_count': 0,
                'mase_values': [],
                'rank_values': [],
                'pattern_counts': defaultdict(int),
            })

            for result in self.results.values():
                all_results = result.get('all_methods_results', {})
                valid_group_results = {
                    method: metrics for method, metrics in all_results.items()
                    if method in group_methods
                    and metrics.get('mase') is not None
                    and np.isfinite(metrics.get('mase'))
                    and metrics.get('mase') < 1e10
                }
                if not valid_group_results:
                    continue

                ranked = sorted(valid_group_results.items(), key=lambda item: item[1]['mase'])
                ranks = {method: rank for rank, (method, _) in enumerate(ranked, start=1)}
                best_method = ranked[0][0]

                for method, metrics in valid_group_results.items():
                    item = stats_by_method[method]
                    item['tested_count'] += 1
                    item['mase_values'].append(metrics['mase'])
                    item['rank_values'].append(ranks[method])
                    item['pattern_counts'][result['pattern']] += 1
                    if method == best_method:
                        item['best_count'] += 1
                    if ranks[method] <= 2:
                        item['top2_count'] += 1

            for method, item in stats_by_method.items():
                tested_count = item['tested_count']
                rows.append({
                    '方法组': group_name,
                    '方法': method,
                    '被组内比较次数': tested_count,
                    '组内第一次数': item['best_count'],
                    '组内第一率(%)': round(item['best_count'] / tested_count * 100, 2) if tested_count else 0,
                    '组内Top2次数': item['top2_count'],
                    '组内Top2率(%)': round(item['top2_count'] / tested_count * 100, 2) if tested_count else 0,
                    '组内平均MASE': round(_mean(item['mase_values']), 4),
                    '组内中位MASE': round(float(np.median(item['mase_values'])), 4) if item['mase_values'] else 0,
                    '组内平均排名': round(_mean(item['rank_values']), 2),
                    '出现模式分布': '; '.join(f"{k}:{v}" for k, v in sorted(item['pattern_counts'].items())),
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                by=['方法组', '组内第一次数', '组内Top2率(%)', '组内平均排名'],
                ascending=[True, False, False, True],
            )
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"方法组对比报告已保存至: {output_path}")

    def save_v6_v61_comparison(self, output_path, v6_results_path):
        try:
            df_v6 = pd.read_csv(v6_results_path)
        except Exception as exc:
            print(f"无法读取V6结果文件: {v6_results_path} ({exc})")
            return

        required_columns = {'MASE', '最优方法', '绝对总量误差'}
        missing_columns = required_columns - set(df_v6.columns)
        if missing_columns:
            print(f"V6结果文件缺少必要列: {', '.join(sorted(missing_columns))}")
            return

        comparison_rows = []
        v61_better_count = 0
        v6_better_count = 0

        for material in self.materials:
            v61_result = self.results[material]

            v6_row = df_v6[df_v6.iloc[:, 0] == material]
            if len(v6_row) == 0:
                continue

            v6_mase = v6_row['MASE'].values[0]
            v6_method = v6_row['最优方法'].values[0]
            v6_error = v6_row['绝对总量误差'].values[0]

            v61_mase = v61_result['mase']
            v61_method = v61_result['best_method']
            v61_error = v61_result['total_error']

            if v61_mase < v6_mase:
                v61_better_count += 1
                winner = 'V6.1'
            elif v61_mase > v6_mase:
                v6_better_count += 1
                winner = 'V6'
            else:
                winner = '持平'

            comparison_rows.append({
                '通用码': material,
                '数据模式': v61_result['pattern'],
                'V6方法': v6_method,
                'V6.1方法': v61_method,
                'V6_MASE': round(v6_mase, 4),
                'V6.1_MASE': round(v61_mase, 4),
                'V6总量误差': round(v6_error, 2),
                'V6.1总量误差': round(v61_error, 2),
                '优胜者': winner,
            })

        df_comparison = pd.DataFrame(comparison_rows)
        df_comparison.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\nV6-V6.1对比报告已保存至: {output_path}")
        print(f"  V6.1优于V6: {v61_better_count} 个物料")
        print(f"  V6优于V6.1: {v6_better_count} 个物料")
        print(f"  持平: {len(comparison_rows) - v61_better_count - v6_better_count} 个物料")


def save_ablation_report(data_path, output_path, baseline_forecaster=None):
    experiments = [
        {
            '实验名称': 'pruned_main_library',
            '禁用方法': [],
        },
    ]

    rows = []
    baseline_summary = baseline_forecaster.summary if baseline_forecaster else None
    baseline_method_counts = baseline_summary['method_distribution'] if baseline_summary else {}

    for experiment in experiments:
        if not experiment['禁用方法'] and baseline_forecaster is not None:
            forecaster = baseline_forecaster
        else:
            forecaster = AdaptiveForecaster(
                data_path,
                test_periods=RuntimeConfig.test_periods,
                disabled_methods=experiment['禁用方法'],
            )
            forecaster.run_analysis()
        summary = forecaster.summary
        method_counts = summary['method_distribution']

        if baseline_summary is None:
            baseline_summary = summary
            baseline_method_counts = method_counts

        rows.append({
            '实验名称': experiment['实验名称'],
            '禁用方法': ', '.join(experiment['禁用方法']) if experiment['禁用方法'] else '',
            '平均MASE': round(summary['avg_mase'], 4),
            '相对基线MASE变化': round(summary['avg_mase'] - baseline_summary['avg_mase'], 4),
            '平均总量误差': round(summary['avg_total_error'], 4),
            '相对基线总量误差变化': round(summary['avg_total_error'] - baseline_summary['avg_total_error'], 4),
            'V6新增方法采用数': summary['v6_new_method_total_uses'],
            'adida胜出数': method_counts.get('adida', 0),
            'winsorized胜出数': method_counts.get('winsorized', 0),
            'weighted_ma_5胜出数': method_counts.get('weighted_ma_5', 0),
            'theta胜出数': method_counts.get('theta', 0),
            'croston胜出数': method_counts.get('croston', 0),
            'optimized_hw胜出数': method_counts.get('optimized_hw', 0),
            '暂移出方法': ', '.join(sorted(TEMPORARILY_REMOVED_METHODS)),
            '删除方法': 'ma_7, holt_winters',
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"消融实验报告已保存至: {output_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    data_path = _first_existing_path(DATA_CANDIDATES)

    try:
        v6_results_path = _first_existing_path(V6_RESULT_CANDIDATES)
    except FileNotFoundError:
        v6_results_path = V6_RESULT_CANDIDATES[0]

    forecaster = AdaptiveForecaster(data_path, test_periods=RuntimeConfig.test_periods)
    forecaster.run_analysis()
    forecaster.print_summary()
    forecaster.save_results('ultimate_prediction_results_v6.1.csv')
    forecaster.save_drift_report('drift_analysis_report_v6.1.csv')
    forecaster.save_method_contribution_report('method_contribution_report_v6.1.csv')
    forecaster.save_method_family_report('method_family_report_v6.1.csv')
    save_ablation_report(data_path, 'ablation_report_v6.1.csv', baseline_forecaster=forecaster)

    print("\n" + "=" * 80)
    print("V6.1预测完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()


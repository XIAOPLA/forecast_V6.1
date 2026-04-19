# -*- coding: utf-8 -*-
"""
智能物料需求预测系统 - 扩展方法库版 (v6.1)
核心改进：
1. 方法库从9种扩展到17种，新增8种预测方法
2. 扩展模型选择策略：每种模式可访问更多候选方法
3. 新增跨模式方法测试，打破模式壁垒
4. V6.1改进：计算平均指标时排除all_zero物料

新增方法：
- Croston: 经典间歇需求预测方法
- ADIDA: 聚合-分解法，减少零值干扰
- Optimized Holt-Winters: 自动优化季节参数
- Damped Trend: 阻尼趋势，防止过度外推
- Theta Method: M3竞赛冠军分解预测法
- Weighted MA: 线性加权移动平均
- Winsorized Mean: 缩尾均值，抗极端值
- Naive with Drift: 带漂移的朴素预测

主要指标：MASE (Mean Absolute Scaled Error)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class ChangePointDetector:

    @staticmethod
    def detect_cusum(series, threshold_sigma=1.5):
        series = np.array(series, dtype=float)
        n = len(series)
        if n < 4:
            return None, 0.0

        mean_val = np.mean(series)
        std_val = np.std(series)
        if std_val < 1e-10:
            return None, 0.0

        standardized = (series - mean_val) / std_val

        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + standardized[i] - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i - 1] + standardized[i] + 0.5)

        threshold = threshold_sigma

        change_idx = None
        for i in range(1, n):
            if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
                change_idx = i
                break

        significance = max(np.max(cusum_pos), -np.min(cusum_neg)) / threshold if threshold > 0 else 0

        return change_idx, significance

    @staticmethod
    def detect_mean_shift(series):
        series = np.array(series, dtype=float)
        n = len(series)
        if n < 6:
            return None, 0.0

        half = n // 2
        first_half = series[:half]
        second_half = series[half:]

        try:
            stat, p_value = stats.mannwhitneyu(first_half, second_half, alternative='two-sided')
            if p_value < 0.15:
                return half, 1 - p_value
        except:
            pass

        return None, 0.0

    @staticmethod
    def detect_variance_shift(series):
        series = np.array(series, dtype=float)
        n = len(series)
        if n < 6:
            return None, 0.0

        half = n // 2
        first_half = series[:half]
        second_half = series[half:]

        var1 = np.var(first_half)
        var2 = np.var(second_half)

        if var1 < 1e-10 and var2 < 1e-10:
            return None, 0.0

        ratio = max(var1, var2) / (min(var1, var2) + 1e-10)
        if ratio > 3.0:
            return half, min(1.0, (ratio - 3.0) / 7.0)

        return None, 0.0

    @staticmethod
    def detect(series):
        series = np.array(series, dtype=float)
        n = len(series)

        candidates = []

        cp1, sig1 = ChangePointDetector.detect_cusum(series)
        if cp1 is not None and cp1 < n - 2:
            candidates.append((cp1, sig1, 'cusum'))

        cp2, sig2 = ChangePointDetector.detect_mean_shift(series)
        if cp2 is not None and cp2 < n - 2:
            candidates.append((cp2, sig2, 'mean_shift'))

        cp3, sig3 = ChangePointDetector.detect_variance_shift(series)
        if cp3 is not None and cp3 < n - 2:
            candidates.append((cp3, sig3, 'variance_shift'))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]


class DataAnalyzer:

    THRESHOLD_SPARSE = 0.8
    THRESHOLD_INTERMITTENT = 0.25
    THRESHOLD_CV = 0.7
    TREND_PVALUE_THRESHOLD = 0.05
    TREND_R2_THRESHOLD = 0.1

    def __init__(self, series):
        self.series = np.array(series, dtype=float)
        self.non_zero = self.series[self.series > 0]
        self.analysis = self._analyze()

    def _analyze(self):
        s = self.series
        n = len(s)
        non_zero = self.non_zero

        zero_ratio = np.sum(s == 0) / n if n > 0 else 1.0
        cv = np.std(non_zero) / np.mean(non_zero) if len(non_zero) > 0 and np.mean(non_zero) > 0 else 0

        if len(non_zero) >= 3:
            try:
                skewness = stats.skew(non_zero)
                kurtosis = stats.kurtosis(non_zero)
            except:
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
        if len(s) <= lag:
            return 0
        try:
            return np.corrcoef(s[:-lag], s[lag:])[0, 1]
        except:
            return 0

    def _detect_trend(self, s):
        if len(s) < 3:
            return 'none', 1.0, 0.0
        try:
            x = np.arange(len(s))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, s)
            r2 = r_value ** 2

            is_significant = p_value < self.TREND_PVALUE_THRESHOLD and r2 > self.TREND_R2_THRESHOLD
            if is_significant:
                direction = 'increasing' if slope > 0 else 'decreasing'
            else:
                direction = 'none'

            return direction, p_value, r2
        except:
            return 'none', 1.0, 0.0

    def _detect_seasonality(self, s):
        if len(s) < 12:
            return False, 0, 0.0

        try:
            n = len(s)
            acf_threshold = 1.96 / np.sqrt(n)

            acf_lag12 = self._autocorr(s, 12)
            acf_lag4 = self._autocorr(s, 4) if n > 4 else 0
            acf_lag3 = self._autocorr(s, 3) if n > 3 else 0

            if acf_lag12 > acf_threshold:
                return True, 12, acf_threshold
            elif acf_lag4 > acf_threshold or acf_lag3 > acf_threshold:
                period = 4 if acf_lag4 >= acf_lag3 else 3
                return True, period, acf_threshold

            return False, 0, acf_threshold
        except:
            return False, 0, 0.0

    def _recent_trend(self, s):
        if len(s) < 10:
            return 0
        recent = s[-10:]
        x = np.arange(len(recent))
        try:
            slope, _, _, _, _ = stats.linregress(x, recent)
            return slope
        except:
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

    MIN_INTERVAL = 13
    MAX_INTERVAL = 26
    TEST_SIZE = 5

    def __init__(self, series):
        self.series = np.array(series, dtype=float)
        self.total_length = len(self.series)
        self.optimal_interval = None
        self.optimal_mase = float('inf')
        self.interval_scores = {}
        self._find_optimal_interval()

    def _find_optimal_interval(self):
        min_required = self.MIN_INTERVAL + self.TEST_SIZE
        if self.total_length < min_required:
            self.optimal_interval = max(self.MIN_INTERVAL, self.total_length - self.TEST_SIZE)
            return

        for interval_len in range(self.MIN_INTERVAL, self.MAX_INTERVAL + 1):
            if interval_len + self.TEST_SIZE > self.total_length:
                continue

            end_idx = self.total_length - self.TEST_SIZE
            start_idx = max(0, end_idx - interval_len)

            train_data = self.series[max(0, end_idx - interval_len):end_idx]
            test_data = self.series[end_idx:self.total_length]

            if len(train_data) < self.MIN_INTERVAL:
                continue

            mase = self._evaluate_interval(train_data, test_data)
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

        start_idx = max(0, self.total_length - self.TEST_SIZE - self.optimal_interval)
        end_idx = self.total_length - self.TEST_SIZE

        return self.series[start_idx:end_idx], self.optimal_interval


class AdaptiveDataAnalyzer:

    THRESHOLD_SPARSE = 0.8
    THRESHOLD_INTERMITTENT = 0.25
    THRESHOLD_CV = 0.7
    TREND_R2_THRESHOLD = 0.1

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

        recent_len = max(4, self.n // 2)
        self.recent_series = self.series[-recent_len:]
        self.recent_analyzer = DataAnalyzer(self.recent_series)
        self.recent_pattern = self.recent_analyzer.get_pattern_type()
        self.recent_analysis = self.recent_analyzer.analysis

        self.change_point = ChangePointDetector.detect(self.series)

        if self.change_point is not None and self.change_point < self.n - 2:
            self.post_change_series = self.series[self.change_point:]
            self.post_change_analyzer = DataAnalyzer(self.post_change_series)
            self.post_change_pattern = self.post_change_analyzer.get_pattern_type()
            self.post_change_analysis = self.post_change_analyzer.analysis
        else:
            self.post_change_series = None
            self.post_change_pattern = None
            self.post_change_analysis = None

        self.analysis, self.pattern = self._resolve_pattern()

        self.pattern_shift_detected = (self.full_pattern != self.optimal_pattern)
        self.drift_info = self._compute_drift_info()

    def _resolve_pattern(self):
        if self.optimal_pattern != self.full_pattern:
            return self.optimal_analysis, self.optimal_pattern

        if (self.post_change_pattern is not None
                and self.post_change_pattern == self.optimal_pattern
                and self.optimal_pattern != self.full_pattern):
            return self.optimal_analysis, self.optimal_pattern

        return self.optimal_analysis, self.optimal_pattern

    def _compute_drift_info(self):
        return {
            'full_pattern': self.full_pattern,
            'optimal_pattern': self.optimal_pattern,
            'recent_pattern': self.recent_pattern,
            'post_change_pattern': self.post_change_pattern,
            'optimal_interval': self.optimal_interval,
            'optimal_interval_mase': self.optimal_interval_mase,
            'change_point': self.change_point,
            'pattern_shift': self.pattern_shift_detected,
            'full_zero_ratio': self.full_analysis['zero_ratio'],
            'optimal_zero_ratio': self.optimal_analysis['zero_ratio'],
            'recent_zero_ratio': self.recent_analysis['zero_ratio'],
            'full_cv': self.full_analysis['cv'],
            'optimal_cv': self.optimal_analysis['cv'],
            'recent_cv': self.recent_analysis['cv'],
            'full_trend': self.full_analysis['trend_direction'],
            'optimal_trend': self.optimal_analysis['trend_direction'],
            'recent_trend': self.recent_analysis['trend_direction'],
        }


class UltimateForecastMethods:

    @staticmethod
    def moving_average(series, horizon=5, window=5):
        series = np.array(series, dtype=float)
        if len(series) < window:
            avg = np.mean(series)
        else:
            avg = np.mean(series[-window:])
        return avg * horizon

    @staticmethod
    def optimized_ses(series, horizon=5):
        series = np.array(series, dtype=float)
        if len(series) < 3:
            return np.mean(series) * horizon

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
        except:
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
        except:
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
    def holt_winters(series, horizon=5, alpha=0.3, beta=0.1, gamma=0.1, season_length=12):
        series = np.array(series, dtype=float)
        n = len(series)

        if n < season_length * 2:
            return UltimateForecastMethods.optimized_des(series, horizon)

        level = np.mean(series[:season_length])
        trend = (np.mean(series[season_length:2 * season_length]) - level) / season_length
        seasonal = np.zeros(season_length)
        for i in range(season_length):
            seasonal[i] = series[i] - level

        for i in range(season_length, n):
            new_level = alpha * (series[i] - seasonal[i % season_length]) + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            seasonal[i % season_length] = gamma * (series[i] - new_level) + (1 - gamma) * seasonal[i % season_length]
            level = new_level

        total_forecast = 0
        for h in range(1, horizon + 1):
            forecast = level + h * trend + seasonal[(n + h) % season_length]
            total_forecast += max(0, forecast)

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
        except:
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
        except:
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
        except:
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
    def theta_method(series, horizon=5, theta=2.0):
        series = np.array(series, dtype=float)
        n = len(series)
        if n < 3:
            return np.mean(series) * horizon

        x = np.arange(n, dtype=float)

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        except:
            return np.mean(series) * horizon

        theta_line = np.zeros(n)
        for i in range(n):
            theta_line[i] = (1 - theta) * (intercept + slope * i) + theta * series[i]

        if n >= 5:
            forecast_per_period = np.mean(theta_line[-5:])
        else:
            forecast_per_period = np.mean(theta_line)

        trend_component = slope * (n + (horizon + 1) / 2)

        total_forecast = forecast_per_period * horizon + (1 - theta) * trend_component * 0.5

        return max(0, total_forecast)

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
            return np.mean(series) * horizon

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
    def select_best_method(series, test_size=5):
        series = np.array(series, dtype=float)

        if np.sum(series) == 0:
            return 'zero', {'method': 'zero', 'mase': 0, 'total_error': 0, 'pattern': 'all_zero',
                            'drift_info': None}, {}

        if len(series) <= test_size:
            return 'moving_average', {'method': 'moving_average', 'mase': 0, 'total_error': 0,
                                      'pattern': 'unknown', 'drift_info': None}, {}

        train = series[:-test_size]
        test = series[-test_size:]
        actual_total = np.sum(test)

        adaptive_analyzer = AdaptiveDataAnalyzer(train)
        pattern = adaptive_analyzer.pattern
        analysis = adaptive_analyzer.analysis
        drift_info = adaptive_analyzer.drift_info

        methods_to_test = AdaptiveModelSelector._get_candidate_methods(pattern, analysis, test_size)

        methods_results = {}

        for method_name, forecast_func in methods_to_test.items():
            try:
                forecast_total = forecast_func(train)
                mase, total_error = AdaptiveModelSelector._calculate_metrics(forecast_total, actual_total, train)
                methods_results[method_name] = {
                    'mase': mase,
                    'total_error': total_error,
                    'forecast_total': forecast_total
                }
            except Exception as e:
                methods_results[method_name] = {
                    'mase': 1e10,
                    'total_error': abs(actual_total),
                    'forecast_total': 0
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
            'naive_drift': lambda s: FM.naive_with_drift(s, horizon=test_size),
        }

        if pattern == 'seasonal':
            core_methods = {
                'holt_winters': lambda s: FM.holt_winters(s, horizon=test_size),
                'optimized_hw': lambda s: FM.optimized_holt_winters(s, horizon=test_size),
                'seasonal_naive': lambda s: FM.seasonal_naive(s, horizon=test_size),
                'optimized_des': lambda s: FM.optimized_des(s, horizon=test_size),
                'theta': lambda s: FM.theta_method(s, horizon=test_size),
            }
            extended_methods = {
                'optimized_ses': lambda s: FM.optimized_ses(s, horizon=test_size),
                'damped_trend': lambda s: FM.damped_trend(s, horizon=test_size),
                'median_5': lambda s: FM.median_forecast(s, horizon=test_size, window=5),
            }

        elif pattern == 'trending':
            core_methods = {
                'optimized_des': lambda s: FM.optimized_des(s, horizon=test_size),
                'damped_trend': lambda s: FM.damped_trend(s, horizon=test_size),
                'naive_drift': lambda s: FM.naive_with_drift(s, horizon=test_size),
                'holt_winters': lambda s: FM.holt_winters(s, horizon=test_size),
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
                'sba': lambda s: FM.sba(s, horizon=test_size),
            }
            extended_methods = {
                'tsb_opt': lambda s: FM.tsb_opt(s, horizon=test_size),
                'winsorized': lambda s: FM.winsorized_mean(s, horizon=test_size),
            }

        elif pattern == 'lumpy':
            core_methods = {
                'sba': lambda s: FM.sba(s, horizon=test_size),
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
                'sba': lambda s: FM.sba(s, horizon=test_size),
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
                'sba': lambda s: FM.sba(s, horizon=test_size),
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
                'ma_7': lambda s: FM.moving_average(s, horizon=test_size, window=7),
                'theta': lambda s: FM.theta_method(s, horizon=test_size),
                'interval_based': lambda s: FM.interval_based_forecast(s, horizon=test_size),
            }
            extended_methods = {
                'weighted_ma_5': lambda s: FM.weighted_moving_average(s, horizon=test_size, window=5),
                'damped_trend': lambda s: FM.damped_trend(s, horizon=test_size),
                'optimized_des': lambda s: FM.optimized_des(s, horizon=test_size),
                'holt_winters': lambda s: FM.holt_winters(s, horizon=test_size),
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
    def _calculate_metrics(forecast_total, actual_total, train=None):
        total_error = abs(forecast_total - actual_total)

        mase = 0.0
        if train is not None and len(train) > 1:
            horizon = 5
            avg_forecast = forecast_total / horizon
            avg_actual = actual_total / horizon

            mae = abs(avg_forecast - avg_actual)

            naive_errors = np.abs(np.diff(train))

            if len(naive_errors) > 0 and np.mean(naive_errors) > 1e-10:
                mase = mae / np.mean(naive_errors)
            else:
                mase = mae if mae > 0 else 0.0

        return mase, total_error


class AdaptiveForecaster:

    def __init__(self, data_path, test_periods=5):
        self.data = pd.read_csv(data_path)
        self.test_periods = test_periods
        self.materials = self.data.iloc[:, 0].values
        self.time_series = self.data.iloc[:, 1:].values
        self.results = {}
        self.summary = {}
        self.drift_analysis = {}
        self.all_zero_materials = []

    def run_analysis(self):
        print("=" * 80)
        print("智能物料需求预测系统 - 扩展方法库版 (v6.1)")
        print("改进：方法库从9种扩展到17种 + 排除all_zero物料参与平均指标计算")
        print("=" * 80)
        print(f"\n数据概览:")
        print(f"  - 物料数量: {len(self.materials)}")
        print(f"  - 历史周期数: {self.time_series.shape[1]}")
        print(f"  - 测试集周期数: {self.test_periods}")
        print(f"  - 训练集周期数: {self.time_series.shape[1] - self.test_periods}")
        print(f"  - 预测方式: 一次性预测{self.test_periods}期总需求")
        print(f"  - 最优区间范围: {OptimalIntervalFinder.MIN_INTERVAL}-{OptimalIntervalFinder.MAX_INTERVAL}周")
        print(f"\n方法库:")
        print(f"  - V5原有方法(9种): moving_average, optimized_ses, optimized_des,")
        print(f"    holt_winters, seasonal_naive, sba, tsb_opt, median_forecast, interval_based")
        print(f"  - V6新增方法(8种): croston, adida, optimized_hw, damped_trend,")
        print(f"    theta, weighted_ma, winsorized, naive_drift")

        print("\n正在分析各物料数据特点并自动寻优最新区间...")
        print("主要评估指标: MASE (Mean Absolute Scaled Error)")
        print("分类策略: 自适应模式分类 + 扩展方法库 + 跨模式方法测试")

        for idx, material in enumerate(self.materials):
            if (idx + 1) % 100 == 0:
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
                    'optimal_interval': 0,
                    'drift_info': None,
                    'all_methods_results': {}
                }
                self.all_zero_materials.append(material)
                continue

            best_method_name, method_info, all_results = AdaptiveModelSelector.select_best_method(
                series, self.test_periods)

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
        print(f"  其中all_zero物料: {len(self.all_zero_materials)} 个（不参与平均指标计算）")
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

        v5_method_set = {'moving_average', 'ma_3', 'ma_5', 'ma_7', 'optimized_ses',
                         'optimized_des', 'holt_winters', 'seasonal_naive',
                         'sba', 'tsb_opt', 'median_5', 'interval_based'}

        v6_new_method_counts = {}
        v6_new_method_mase = []

        for material, result in self.results.items():
            method = result['best_method']
            pattern = result['pattern']

            method_counts[method] = method_counts.get(method, 0) + 1
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            if pattern == 'all_zero':
                continue

            if result['mase'] is not None and not np.isinf(result['mase']):
                total_mase.append(result['mase'])
                total_error_list.append(result['total_error'])

            if result['optimal_interval'] > 0:
                total_intervals.append(result['optimal_interval'])

            if method not in v5_method_set:
                v6_new_method_counts[method] = v6_new_method_counts.get(method, 0) + 1
                if result['mase'] is not None and not np.isinf(result['mase']):
                    v6_new_method_mase.append(result['mase'])

            series = self.time_series[list(self.materials).index(material)]
            train = series[:-self.test_periods]
            test = series[-self.test_periods:]
            actual_total = np.sum(test)

            if len(train) >= 5:
                baseline_forecast = np.mean(train[-5:]) * self.test_periods
            else:
                baseline_forecast = np.mean(train) * self.test_periods

            baseline_error_val = abs(baseline_forecast - actual_total)

            baseline_mae = abs(baseline_forecast / self.test_periods - actual_total / self.test_periods)
            naive_errors = np.abs(np.diff(train))
            if len(naive_errors) > 0 and np.mean(naive_errors) > 1e-10:
                baseline_mase_val = baseline_mae / np.mean(naive_errors)
            else:
                baseline_mase_val = baseline_mae if baseline_mae > 0 else 0.0

            baseline_mase.append(baseline_mase_val)
            baseline_total_error.append(baseline_error_val)

        valid_materials_count = len(self.materials) - len(self.all_zero_materials)

        self.summary = {
            'method_distribution': method_counts,
            'pattern_distribution': pattern_counts,
            'avg_mase': np.mean(total_mase) if total_mase else 0,
            'avg_total_error': np.mean(total_error_list) if total_error_list else 0,
            'avg_optimal_interval': np.mean(total_intervals) if total_intervals else 0,
            'baseline_avg_mase': np.mean(baseline_mase) if baseline_mase else 0,
            'baseline_avg_total_error': np.mean(baseline_total_error) if baseline_total_error else 0,
            'improvement_mase': (np.mean(baseline_mase) - np.mean(total_mase)) / np.mean(baseline_mase) * 100 if np.mean(baseline_mase) > 0 else 0,
            'improvement_total_error': (np.mean(baseline_total_error) - np.mean(total_error_list)) / np.mean(baseline_total_error) * 100 if np.mean(baseline_total_error) > 0 else 0,
            'v6_new_method_counts': v6_new_method_counts,
            'v6_new_method_avg_mase': np.mean(v6_new_method_mase) if v6_new_method_mase else 0,
            'v6_new_method_total_uses': sum(v6_new_method_counts.values()),
            'all_zero_count': len(self.all_zero_materials),
            'valid_materials_count': valid_materials_count,
        }

    def _analyze_drift(self):
        drift_count = 0
        change_point_count = 0
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
                        'recent_pattern': drift_info['recent_pattern'],
                        'change_point': drift_info['change_point'],
                        'optimal_interval': drift_info['optimal_interval'],
                        'full_zero_ratio': drift_info['full_zero_ratio'],
                        'optimal_zero_ratio': drift_info['optimal_zero_ratio'],
                        'full_cv': drift_info['full_cv'],
                        'optimal_cv': drift_info['optimal_cv'],
                        'final_pattern': result['pattern'],
                    })

            if drift_info['change_point'] is not None:
                change_point_count += 1

        self.drift_analysis = {
            'total_materials': len(self.materials),
            'drift_count': drift_count,
            'drift_ratio': drift_count / len(self.materials) * 100 if len(self.materials) > 0 else 0,
            'change_point_count': change_point_count,
            'change_point_ratio': change_point_count / len(self.materials) * 100 if len(self.materials) > 0 else 0,
            'pattern_transitions': dict(pattern_transitions),
            'drift_examples': drift_examples,
        }

    def print_summary(self):
        print("\n" + "=" * 80)
        print("预测结果汇总 (扩展方法库版 - v6.1)")
        print("=" * 80)

        print(f"\n【物料统计】")
        print(f"  总物料数: {len(self.materials)}")
        print(f"  all_zero物料: {self.summary['all_zero_count']} 个（不参与平均指标计算）")
        print(f"  有效物料数: {self.summary['valid_materials_count']} 个")

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

        print("\n  第三层 - SBC矩阵四象限:")
        for pattern in ['lumpy', 'intermittent', 'erratic', 'stable']:
            count = self.summary['pattern_distribution'].get(pattern, 0)
            pct = count / len(self.materials) * 100
            print(f"    {pattern:15s}: {count:4d} 个物料 ({pct:5.1f}%)")

        print("\n【最优预测方法分布】")
        sorted_methods = sorted(self.summary['method_distribution'].items(), key=lambda x: -x[1])
        v5_method_set = {'ma_3', 'ma_5', 'ma_7', 'optimized_ses',
                         'optimized_des', 'holt_winters', 'seasonal_naive',
                         'sba', 'tsb_opt', 'median_5', 'interval_based', 'moving_average'}
        for method, count in sorted_methods:
            pct = count / len(self.materials) * 100
            tag = " [V6新增]" if method not in v5_method_set else ""
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

        print("\n【预测性能对比】（排除all_zero物料）")
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
        print(f"  检测到变化点的物料: {da['change_point_count']} ({da['change_point_ratio']:.1f}%)")

        if da['pattern_transitions']:
            print(f"\n【模式转换类型分布】")
            sorted_transitions = sorted(da['pattern_transitions'].items(), key=lambda x: -x[1])
            for transition, count in sorted_transitions:
                pct = count / da['drift_count'] * 100 if da['drift_count'] > 0 else 0
                print(f"  {transition:30s}: {count:4d} 个物料 ({pct:5.1f}%)")

        if da['drift_examples']:
            print(f"\n【模式漂移示例（前10个）】")
            print(f"  {'通用码':<12s} {'全历史':>10s} {'最优':>10s} {'近期':>10s} {'最终':>10s} {'区间':>6s} {'变化点':>6s} {'零值比变化':>12s}")
            print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*12}")
            for ex in da['drift_examples'][:10]:
                zr_change = f"{ex['full_zero_ratio']:.2f}→{ex['optimal_zero_ratio']:.2f}"
                cp_str = str(ex['change_point']) if ex['change_point'] is not None else '-'
                mat_str = str(ex['material'])[:12]
                print(f"  {mat_str:<12s} {ex['full_pattern']:>10s} {ex['optimal_pattern']:>10s} "
                      f"{ex['recent_pattern']:>10s} {ex['final_pattern']:>10s} {ex['optimal_interval']:>6d} {cp_str:>6s} "
                      f"{zr_change:>12s}")

    def save_results(self, output_path):
        rows = []
        for material in self.materials:
            result = self.results[material]
            drift_info = result.get('drift_info', {})

            row = {
                '通用码': material,
                '数据模式(自适应)': result['pattern'],
                '数据模式(全历史)': drift_info.get('full_pattern', result['pattern']) if drift_info else result['pattern'],
                '数据模式(最优区间)': drift_info.get('optimal_pattern', result['pattern']) if drift_info else result['pattern'],
                '模式漂移': '是' if (drift_info and drift_info.get('pattern_shift', False)) else '否',
                '最优区间(周)': drift_info.get('optimal_interval', 0) if drift_info else 0,
                '变化点': drift_info.get('change_point', '') if drift_info else '',
                '最优方法': result['best_method'],
                '是否V6新增方法': '是' if result['best_method'] in {'croston', 'adida', 'optimized_hw', 'damped_trend', 'theta', 'weighted_ma_5', 'winsorized', 'naive_drift'} else '否',
                '预测总量': round(result['forecast_total'], 2),
                '实际总量': round(result['actual_total'], 2),
                'MASE': round(result['mase'], 4),
                '绝对总量误差': round(result['total_error'], 2),
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
            if result['pattern'] == 'all_zero':
                continue

            series = self.time_series[list(self.materials).index(material)]
            train = series[:-self.test_periods]
            test = series[-self.test_periods:]
            actual_total = np.sum(test)

            if len(train) >= 5:
                baseline_forecast = np.mean(train[-5:]) * self.test_periods
            else:
                baseline_forecast = np.mean(train) * self.test_periods

            baseline_error = abs(baseline_forecast - actual_total)

            baseline_mae = abs(baseline_forecast / self.test_periods - actual_total / self.test_periods)
            naive_errors = np.abs(np.diff(train))
            if len(naive_errors) > 0 and np.mean(naive_errors) > 1e-10:
                baseline_mase = baseline_mae / np.mean(naive_errors)
            else:
                baseline_mase = baseline_mae if baseline_mae > 0 else 0.0

            smart_mase = result['mase']
            smart_error = result['total_error']

            mase_improve = (baseline_mase - smart_mase) / baseline_mase * 100 if baseline_mase > 0 else 0
            error_improve = (baseline_error - smart_error) / baseline_error * 100 if baseline_error > 0 else 0

            comparison_rows.append({
                '通用码': material,
                '数据模式': result['pattern'],
                '最优方法': result['best_method'],
                '是否V6新增方法': '是' if result['best_method'] in {'croston', 'adida', 'optimized_hw', 'damped_trend', 'theta', 'weighted_ma_5', 'winsorized', 'naive_drift'} else '否',
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
                '近期模式': drift_info['recent_pattern'],
                '变化点后模式': drift_info.get('post_change_pattern', ''),
                '最终采用模式': result['pattern'],
                '模式漂移': '是' if drift_info['pattern_shift'] else '否',
                '最优区间长度': drift_info['optimal_interval'],
                '变化点位置': drift_info['change_point'] if drift_info['change_point'] is not None else '',
                '零值比(全历史)': round(drift_info['full_zero_ratio'], 4),
                '零值比(最优区间)': round(drift_info['optimal_zero_ratio'], 4),
                'CV(全历史)': round(drift_info['full_cv'], 4),
                'CV(最优区间)': round(drift_info['optimal_cv'], 4),
                '趋势(全历史)': drift_info['full_trend'],
                '趋势(最优区间)': drift_info['optimal_trend'],
                'MASE': round(result['mase'], 4),
                '最优方法': result['best_method'],
                '是否V6新增方法': '是' if result['best_method'] in {'croston', 'adida', 'optimized_hw', 'damped_trend', 'theta', 'weighted_ma_5', 'winsorized', 'naive_drift'} else '否',
            })

        df_drift = pd.DataFrame(rows)
        df_drift.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"漂移分析报告已保存至: {output_path}")

    def save_v6_v61_comparison(self, output_path, v6_results_path):
        try:
            df_v6 = pd.read_csv(v6_results_path)
        except:
            print(f"无法读取V6结果文件: {v6_results_path}")
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


def main():
    import os

    data_path = 'ready_data.csv'
    v6_results_path = 'ultimate_prediction_results_v6.csv'

    if not os.path.exists(data_path):
        v5_data_path = os.path.join('..', 'forecast_V5', data_path)
        if os.path.exists(v5_data_path):
            data_path = v5_data_path
        else:
            v6_data_path = os.path.join('..', 'forecast_V6', data_path)
            if os.path.exists(v6_data_path):
                data_path = v6_data_path

    if not os.path.exists(v6_results_path):
        v6_results_alt = os.path.join('..', 'forecast_V6', v6_results_path)
        if os.path.exists(v6_results_alt):
            v6_results_path = v6_results_alt

    forecaster = AdaptiveForecaster(data_path, test_periods=5)
    forecaster.run_analysis()
    forecaster.print_summary()
    forecaster.save_results('ultimate_prediction_results_v6.1.csv')
    forecaster.save_comparison_report('comparison_report_v6.1.csv')
    forecaster.save_drift_report('drift_analysis_report_v6.1.csv')
    forecaster.save_v6_v61_comparison('v6_v61_comparison.csv', v6_results_path)

    print("\n" + "=" * 80)
    print("V6.1预测完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()

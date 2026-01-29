"""
SERE - Sustainable Exergy Risk Engine
Advanced risk control system with Monte Carlo, VaR, and CVaR
Version 2.0 - Production Ready

Core Philosophy:
- No point forecasts - all decisions based on distributions
- Tail risk explicitly drives decisions
- CVaR (not expected value) is primary control metric
- Resilience and capital protection over profit maximization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import norm, multivariate_normal
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# 1. CORE DATA STRUCTURES
# ================================================================

class RiskState(Enum):
    """Risk state classification"""
    HOLD = "HOLD"           # Within risk appetite, no action needed
    WATCH = "WATCH"         # Approaching limits, monitor closely
    ACTION = "ACTION"       # Risk appetite breached, action required
    CRITICAL = "CRITICAL"   # Extreme risk, immediate intervention


class ScenarioType(Enum):
    """Portfolio scenario classification"""
    BALANCED = "BALANCED"
    SURPLUS = "SURPLUS"
    SHORTAGE = "SHORTAGE"
    EXTREME_SHORTAGE = "EXTREME_SHORTAGE"
    EXTREME_SURPLUS = "EXTREME_SURPLUS"


class ActionType(Enum):
    """Available mitigation actions"""
    DO_NOTHING = "DO_NOTHING"
    INTRADAY_BUY = "INTRADAY_BUY"
    INTRADAY_SELL = "INTRADAY_SELL"
    ACTIVATE_DEMAND_FLEX = "ACTIVATE_DEMAND_FLEX"
    CURTAIL_PV = "CURTAIL_PV"
    CURTAIL_WIND = "CURTAIL_WIND"
    STORAGE_CHARGE = "STORAGE_CHARGE"
    STORAGE_DISCHARGE = "STORAGE_DISCHARGE"
    LOAD_SHIFT = "LOAD_SHIFT"


@dataclass
class UncertaintyInput:
    """Probabilistic input with full distribution parameters"""
    name: str
    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    distribution_type: str = "normal"  # normal, lognormal, triangular
    
    @property
    def uncertainty_range(self) -> float:
        return self.p90 - self.p10
    
    @property
    def coefficient_of_variation(self) -> float:
        if self.mean == 0:
            return float('inf')
        return self.std / abs(self.mean)


@dataclass
class CorrelationMatrix:
    """Correlation structure between uncertainties"""
    demand_pv: float = -0.3      # Demand-PV anticorrelation (sunny = less demand)
    demand_wind: float = 0.1     # Demand-Wind weak correlation
    pv_wind: float = -0.2        # PV-Wind anticorrelation
    
    def get_matrix(self) -> np.ndarray:
        """Return 3x3 correlation matrix [Demand, PV, Wind]"""
        return np.array([
            [1.0, self.demand_pv, self.demand_wind],
            [self.demand_pv, 1.0, self.pv_wind],
            [self.demand_wind, self.pv_wind, 1.0]
        ])


@dataclass
class FlexibilityAsset:
    """Flexibility resource specification"""
    name: str
    max_mw: float
    duration_minutes: int
    cost_per_mwh: float
    response_time_minutes: int = 5
    recovery_time_minutes: int = 60
    available: bool = True
    current_usage_mw: float = 0.0
    activation_count_today: int = 0
    max_activations_per_day: int = 10
    
    @property
    def remaining_capacity(self) -> float:
        return self.max_mw - self.current_usage_mw
    
    @property
    def can_activate(self) -> bool:
        return (self.available and 
                self.remaining_capacity > 0 and 
                self.activation_count_today < self.max_activations_per_day)


@dataclass
class StorageAsset:
    """Battery/storage specification"""
    capacity_mwh: float
    max_power_mw: float
    efficiency: float
    cost_per_mwh: float
    degradation_cost_per_cycle: float = 0.0
    current_soc: float = 0.5
    min_soc: float = 0.1
    max_soc: float = 0.9
    
    @property
    def available_discharge_mwh(self) -> float:
        return self.capacity_mwh * (self.current_soc - self.min_soc)
    
    @property
    def available_charge_mwh(self) -> float:
        return self.capacity_mwh * (self.max_soc - self.current_soc)
    
    @property
    def available_discharge_mw(self) -> float:
        return min(self.available_discharge_mwh * 4, self.max_power_mw)  # 15-min = 0.25h
    
    @property
    def available_charge_mw(self) -> float:
        return min(self.available_charge_mwh * 4, self.max_power_mw)


@dataclass
class MarketPrices:
    """Market price structure with distributions"""
    day_ahead_mean: float
    day_ahead_std: float
    
    intraday_bid_mean: float
    intraday_bid_std: float
    intraday_ask_mean: float
    intraday_ask_std: float
    
    rebap_plus_mean: float        # Shortage penalty
    rebap_plus_std: float
    rebap_plus_p95: float         # 95th percentile
    
    rebap_minus_mean: float       # Surplus penalty
    rebap_minus_std: float
    rebap_minus_p95: float
    
    liquidity_indicator: float = 1.0  # 0-1, affects intraday execution


@dataclass
class RiskAppetite:
    """Governance & risk limits"""
    # Primary risk controls
    cvar_limit_eur: float                    # Maximum CVaR per interval
    var_limit_eur: float                     # Maximum VaR per interval
    confidence_level: float = 0.95           # 95% or 99%
    
    # Daily aggregated limits
    daily_cvar_limit_eur: float = None
    max_rebap_exposure_mw: float = 10.0
    
    # Risk aversion parameter
    lambda_risk_aversion: float = 2.0
    
    # Decision thresholds
    hold_threshold_mw: float = 2.0
    watch_threshold_mw: float = 5.0
    action_threshold_mw: float = 10.0
    
    # Escalation rules
    manual_override_threshold_eur: float = 5000.0
    automated_action_max_eur: float = 2000.0
    
    def __post_init__(self):
        if self.daily_cvar_limit_eur is None:
            self.daily_cvar_limit_eur = self.cvar_limit_eur * 96  # 96 intervals per day


@dataclass
class PortfolioState:
    """Complete portfolio state at time t"""
    timestamp: datetime
    
    # Uncertainty inputs
    demand: UncertaintyInput
    pv_generation: UncertaintyInput
    wind_generation: UncertaintyInput
    
    # Correlation structure
    correlations: CorrelationMatrix
    
    # Current hedge position
    hedge_position_mw: float
    
    # Available resources
    demand_flexibility: List[FlexibilityAsset]
    storage: Optional[StorageAsset]
    pv_curtailment_available_mw: float
    wind_curtailment_available_mw: float
    
    # Market conditions
    market_prices: MarketPrices
    
    # Governance
    risk_appetite: RiskAppetite
    
    # Optional fields with defaults
    curtailment_compensation_rate: float = 0.0  # 0-1
    gate_closure_minutes: int = 5
    negative_price_compensation: bool = True
    regulatory_flags: Dict[str, Any] = field(default_factory=dict)


# ================================================================
# 2. MONTE CARLO SIMULATION ENGINE
# ================================================================

class MonteCarloEngine:
    """
    Advanced Monte Carlo simulation with correlation modeling
    Generates ≥10,000 scenarios for loss distribution
    """
    
    def __init__(self, n_scenarios: int = 10000, random_seed: Optional[int] = None):
        self.n_scenarios = n_scenarios
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_correlated_scenarios(self, 
                                     state: PortfolioState) -> Dict[str, np.ndarray]:
        """
        Generate correlated scenarios for demand, PV, wind
        Returns: dict with 'demand', 'pv', 'wind' arrays
        """
        # Build mean vector and covariance matrix
        means = np.array([
            state.demand.mean,
            state.pv_generation.mean,
            state.wind_generation.mean
        ])
        
        stds = np.array([
            state.demand.std,
            state.pv_generation.std,
            state.wind_generation.std
        ])
        
        # Correlation matrix
        corr_matrix = state.correlations.get_matrix()
        
        # Covariance matrix = D * R * D (D = diagonal std matrix)
        cov_matrix = np.outer(stds, stds) * corr_matrix
        
        # Generate multivariate normal samples
        samples = multivariate_normal.rvs(
            mean=means,
            cov=cov_matrix,
            size=self.n_scenarios,
            random_state=self.random_seed
        )
        
        # Ensure 2D array even for single sample
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        
        # Apply non-negativity constraints for generation
        scenarios = {
            'demand': np.maximum(samples[:, 0], 0),
            'pv': np.maximum(samples[:, 1], 0),
            'wind': np.maximum(samples[:, 2], 0)
        }
        
        return scenarios
    
    def calculate_net_load_scenarios(self, 
                                     scenarios: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate net load for all scenarios
        Net Load = Demand - PV - Wind
        """
        net_load = (scenarios['demand'] - 
                   scenarios['pv'] - 
                   scenarios['wind'])
        return net_load
    
    def calculate_residual_scenarios(self,
                                     net_load: np.ndarray,
                                     hedge_mw: float) -> np.ndarray:
        """
        Calculate residual position
        Residual = Net Load - Hedge
        """
        return net_load - hedge_mw
    
    def calculate_loss_distribution(self,
                                    residual: np.ndarray,
                                    prices: MarketPrices) -> np.ndarray:
        """
        Calculate financial loss for each scenario
        Loss = positive for costs, negative for revenues
        """
        # Generate price scenarios
        rebap_plus_scenarios = np.random.normal(
            prices.rebap_plus_mean,
            prices.rebap_plus_std,
            self.n_scenarios
        )
        rebap_plus_scenarios = np.maximum(rebap_plus_scenarios, 0)
        
        rebap_minus_scenarios = np.random.normal(
            prices.rebap_minus_mean,
            prices.rebap_minus_std,
            self.n_scenarios
        )
        
        # Calculate losses
        # Shortage (positive residual) = cost at ReBAP+
        # Surplus (negative residual) = revenue loss at ReBAP-
        losses = np.where(
            residual > 0,
            residual * rebap_plus_scenarios * 0.25,  # 15-min in hours
            -residual * np.abs(rebap_minus_scenarios) * 0.25
        )
        
        return losses
    
    def run_full_simulation(self, state: PortfolioState) -> Dict[str, Any]:
        """
        Execute complete Monte Carlo simulation
        Returns comprehensive results package
        """
        # Generate scenarios
        scenarios = self.generate_correlated_scenarios(state)
        
        # Calculate net load
        net_load = self.calculate_net_load_scenarios(scenarios)
        
        # Calculate residual
        residual = self.calculate_residual_scenarios(net_load, state.hedge_position_mw)
        
        # Calculate losses
        losses = self.calculate_loss_distribution(residual, state.market_prices)
        
        return {
            'scenarios': scenarios,
            'net_load': net_load,
            'residual': residual,
            'losses': losses,
            'n_scenarios': self.n_scenarios
        }


# ================================================================
# 3. RISK METRICS CALCULATOR
# ================================================================

class RiskMetricsCalculator:
    """Calculate VaR, CVaR, and all risk metrics"""
    
    @staticmethod
    def calculate_var(losses: np.ndarray, confidence: float = 0.95) -> float:
        """
        Value at Risk (VaR) at confidence level
        VaR = α-percentile of loss distribution
        """
        return np.percentile(losses, confidence * 100)
    
    @staticmethod
    def calculate_cvar(losses: np.ndarray, confidence: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR)
        CVaR = Expected loss in worst (1-α)% of scenarios
        """
        var = np.percentile(losses, confidence * 100)
        tail_losses = losses[losses >= var]
        
        if len(tail_losses) == 0:
            return var
        
        cvar = np.mean(tail_losses)
        return cvar
    
    @staticmethod
    def calculate_expected_shortfall(residual: np.ndarray, 
                                     percentile: float = 95) -> float:
        """Expected Shortfall (ES) - magnitude of shortage in tail"""
        threshold = np.percentile(residual, percentile)
        tail_residuals = residual[residual >= threshold]
        
        if len(tail_residuals) == 0:
            return 0.0
        
        return np.mean(tail_residuals)
    
    @staticmethod
    def calculate_tail_probability(residual: np.ndarray, 
                                   threshold_mw: float) -> float:
        """Probability of exceeding threshold"""
        return np.mean(residual > threshold_mw)
    
    @staticmethod
    def identify_tail_scenarios(losses: np.ndarray,
                               residual: np.ndarray,
                               scenarios: Dict[str, np.ndarray],
                               confidence: float = 0.95) -> Dict[str, Any]:
        """
        Identify and characterize tail scenarios
        Returns statistics about worst outcomes
        """
        var = np.percentile(losses, confidence * 100)
        tail_indices = losses >= var
        
        tail_stats = {
            'count': np.sum(tail_indices),
            'mean_loss': np.mean(losses[tail_indices]),
            'max_loss': np.max(losses[tail_indices]),
            'mean_residual': np.mean(residual[tail_indices]),
            'max_residual': np.max(residual[tail_indices]),
            'mean_demand': np.mean(scenarios['demand'][tail_indices]),
            'mean_pv': np.mean(scenarios['pv'][tail_indices]),
            'mean_wind': np.mean(scenarios['wind'][tail_indices])
        }
        
        return tail_stats
    
    @classmethod
    def calculate_all_metrics(cls,
                             simulation_results: Dict[str, Any],
                             state: PortfolioState) -> 'RiskMetrics':
        """Calculate complete risk metrics package"""
        
        losses = simulation_results['losses']
        residual = simulation_results['residual']
        scenarios = simulation_results['scenarios']
        net_load = simulation_results['net_load']
        
        # Core risk metrics
        expected_loss = np.mean(losses)
        var_95 = cls.calculate_var(losses, state.risk_appetite.confidence_level)
        cvar_95 = cls.calculate_cvar(losses, state.risk_appetite.confidence_level)
        
        # Tail analysis
        tail_stats = cls.identify_tail_scenarios(
            losses, residual, scenarios, state.risk_appetite.confidence_level
        )
        
        # Probabilities
        prob_shortage = np.mean(residual > 0)
        prob_surplus = np.mean(residual < 0)
        prob_extreme_shortage = cls.calculate_tail_probability(
            residual, state.risk_appetite.action_threshold_mw
        )
        
        # Exposure metrics
        short_exposure_mean = np.mean(np.maximum(residual, 0))
        surplus_exposure_mean = np.mean(np.maximum(-residual, 0))
        
        # Risk Energy (composite metric)
        risk_energy = expected_loss + state.risk_appetite.lambda_risk_aversion * cvar_95
        
        # Distribution statistics
        residual_p10 = np.percentile(residual, 10)
        residual_p50 = np.percentile(residual, 50)
        residual_p90 = np.percentile(residual, 90)
        
        # Confidence score (inverse of relative uncertainty)
        uncertainty_range = residual_p90 - residual_p10
        if abs(residual_p50) > 0.1:
            confidence_pct = 100 * (1 - uncertainty_range / (2 * abs(residual_p50)))
        else:
            confidence_pct = 50.0
        confidence_pct = np.clip(confidence_pct, 0, 100)
        
        # Classify scenario and risk state
        scenario_type = cls._classify_scenario(residual_p50, residual_p90, 
                                              prob_extreme_shortage, state.risk_appetite)
        risk_state = cls._classify_risk_state(cvar_95, var_95, confidence_pct, 
                                              state.risk_appetite)
        
        return RiskMetrics(
            timestamp=state.timestamp,
            n_scenarios=simulation_results['n_scenarios'],
            expected_loss_eur=expected_loss,
            var_95_eur=var_95,
            cvar_95_eur=cvar_95,
            risk_energy_eur=risk_energy,
            expected_net_load_mw=np.mean(net_load),
            residual_p10_mw=residual_p10,
            residual_p50_mw=residual_p50,
            residual_p90_mw=residual_p90,
            short_exposure_mw=short_exposure_mean,
            surplus_exposure_mw=surplus_exposure_mean,
            prob_shortage=prob_shortage,
            prob_surplus=prob_surplus,
            prob_extreme_shortage=prob_extreme_shortage,
            tail_statistics=tail_stats,
            confidence_pct=confidence_pct,
            scenario_type=scenario_type,
            risk_state=risk_state
        )
    
    @staticmethod
    def _classify_scenario(residual_p50: float,
                          residual_p90: float,
                          prob_extreme: float,
                          risk_appetite: RiskAppetite) -> ScenarioType:
        """Classify portfolio scenario"""
        if abs(residual_p50) <= risk_appetite.hold_threshold_mw:
            return ScenarioType.BALANCED
        
        if residual_p90 > risk_appetite.action_threshold_mw or prob_extreme > 0.2:
            return ScenarioType.EXTREME_SHORTAGE
        
        if residual_p50 > risk_appetite.watch_threshold_mw:
            return ScenarioType.SHORTAGE
        
        if residual_p50 < -risk_appetite.action_threshold_mw:
            return ScenarioType.EXTREME_SURPLUS
        
        return ScenarioType.SURPLUS
    
    @staticmethod
    def _classify_risk_state(cvar: float,
                            var: float,
                            confidence: float,
                            risk_appetite: RiskAppetite) -> RiskState:
        """Classify risk state"""
        if cvar > risk_appetite.cvar_limit_eur * 1.5:
            return RiskState.CRITICAL
        
        if cvar > risk_appetite.cvar_limit_eur:
            return RiskState.ACTION
        
        if cvar > risk_appetite.cvar_limit_eur * 0.75 or var > risk_appetite.var_limit_eur:
            return RiskState.WATCH
        
        return RiskState.HOLD


@dataclass
class RiskMetrics:
    """Complete risk metrics output"""
    timestamp: datetime
    n_scenarios: int
    
    # Core risk metrics
    expected_loss_eur: float
    var_95_eur: float
    cvar_95_eur: float
    risk_energy_eur: float
    
    # Position metrics
    expected_net_load_mw: float
    residual_p10_mw: float
    residual_p50_mw: float
    residual_p90_mw: float
    short_exposure_mw: float
    surplus_exposure_mw: float
    
    # Probabilities
    prob_shortage: float
    prob_surplus: float
    prob_extreme_shortage: float
    
    # Tail analysis
    tail_statistics: Dict[str, Any]
    
    # Classification
    confidence_pct: float
    scenario_type: ScenarioType
    risk_state: RiskState


# ================================================================
# 4. MITIGATION ACTION OPTIMIZER
# ================================================================

@dataclass
class MitigationAction:
    """Single mitigation action evaluation"""
    action_type: ActionType
    volume_mw: float
    cost_eur: float
    marginal_cost_eur_per_mwh: float
    
    # Impact metrics
    cvar_reduction_eur: float
    var_reduction_eur: float
    residual_after_mw: float
    
    # Feasibility
    feasible: bool
    operational_constraints: List[str]
    
    # Explainability
    rationale: str
    assumptions: List[str]


class MitigationOptimizer:
    """Evaluate and rank mitigation actions to reduce CVaR"""
    
    def __init__(self, mc_engine: MonteCarloEngine):
        self.mc_engine = mc_engine
    
    def evaluate_do_nothing(self,
                           state: PortfolioState,
                           risk_metrics: RiskMetrics) -> MitigationAction:
        """Baseline: accept ReBAP exposure"""
        return MitigationAction(
            action_type=ActionType.DO_NOTHING,
            volume_mw=0,
            cost_eur=risk_metrics.expected_loss_eur,
            marginal_cost_eur_per_mwh=float('inf'),
            cvar_reduction_eur=0,
            var_reduction_eur=0,
            residual_after_mw=risk_metrics.residual_p50_mw,
            feasible=True,
            operational_constraints=[],
            rationale="Accept current risk position, no mitigation",
            assumptions=["Market conditions remain stable", "No operational disruptions"]
        )
    
    def evaluate_intraday_buy(self,
                             state: PortfolioState,
                             current_metrics: RiskMetrics,
                             volume_mw: float) -> MitigationAction:
        """Evaluate intraday purchase"""
        
        # Calculate cost
        cost = volume_mw * state.market_prices.intraday_bid_mean * 0.25
        marginal_cost = state.market_prices.intraday_bid_mean
        
        # Simulate with adjusted hedge
        adjusted_state = self._adjust_hedge(state, volume_mw)
        sim_results = self.mc_engine.run_full_simulation(adjusted_state)
        new_metrics = RiskMetricsCalculator.calculate_all_metrics(sim_results, adjusted_state)
        
        # Calculate reductions
        cvar_reduction = current_metrics.cvar_95_eur - new_metrics.cvar_95_eur
        var_reduction = current_metrics.var_95_eur - new_metrics.var_95_eur
        
        # Check feasibility
        constraints = []
        feasible = True
        
        if cost > state.risk_appetite.automated_action_max_eur:
            constraints.append("Exceeds automated action limit - manual approval required")
            feasible = False
        
        if state.market_prices.liquidity_indicator < 0.5:
            constraints.append("Low market liquidity")
        
        return MitigationAction(
            action_type=ActionType.INTRADAY_BUY,
            volume_mw=volume_mw,
            cost_eur=cost,
            marginal_cost_eur_per_mwh=marginal_cost,
            cvar_reduction_eur=cvar_reduction,
            var_reduction_eur=var_reduction,
            residual_after_mw=new_metrics.residual_p50_mw,
            feasible=feasible,
            operational_constraints=constraints,
            rationale=f"Buy {volume_mw:.1f} MW intraday to reduce shortage risk",
            assumptions=[
                f"Intraday price: {marginal_cost:.2f} €/MWh",
                f"Liquidity: {state.market_prices.liquidity_indicator:.1%}",
                f"Gate closure: {state.gate_closure_minutes} min"
            ]
        )
    
    def evaluate_flexibility(self,
                            state: PortfolioState,
                            current_metrics: RiskMetrics,
                            asset: FlexibilityAsset,
                            volume_mw: float) -> MitigationAction:
        """Evaluate demand flexibility activation"""
        
        # Check feasibility
        constraints = []
        feasible = asset.can_activate
        
        if volume_mw > asset.remaining_capacity:
            constraints.append(f"Volume exceeds available capacity ({asset.remaining_capacity:.1f} MW)")
            feasible = False
            volume_mw = min(volume_mw, asset.remaining_capacity)
        
        if asset.response_time_minutes > state.gate_closure_minutes:
            constraints.append(f"Response time ({asset.response_time_minutes} min) > gate closure")
            feasible = False
        
        # Calculate cost
        cost = volume_mw * asset.cost_per_mwh * 0.25
        
        # Simulate with flexibility
        adjusted_state = self._adjust_hedge(state, volume_mw)
        sim_results = self.mc_engine.run_full_simulation(adjusted_state)
        new_metrics = RiskMetricsCalculator.calculate_all_metrics(sim_results, adjusted_state)
        
        cvar_reduction = current_metrics.cvar_95_eur - new_metrics.cvar_95_eur
        var_reduction = current_metrics.var_95_eur - new_metrics.var_95_eur
        
        return MitigationAction(
            action_type=ActionType.ACTIVATE_DEMAND_FLEX,
            volume_mw=volume_mw,
            cost_eur=cost,
            marginal_cost_eur_per_mwh=asset.cost_per_mwh,
            cvar_reduction_eur=cvar_reduction,
            var_reduction_eur=var_reduction,
            residual_after_mw=new_metrics.residual_p50_mw,
            feasible=feasible,
            operational_constraints=constraints,
            rationale=f"Activate {asset.name}: {volume_mw:.1f} MW",
            assumptions=[
                f"Asset available: {asset.remaining_capacity:.1f} MW",
                f"Activations today: {asset.activation_count_today}/{asset.max_activations_per_day}",
                f"Recovery time: {asset.recovery_time_minutes} min"
            ]
        )
    
    def evaluate_storage_discharge(self,
                                   state: PortfolioState,
                                   current_metrics: RiskMetrics,
                                   volume_mw: float) -> MitigationAction:
        """Evaluate battery discharge"""
        
        if state.storage is None:
            return MitigationAction(
                action_type=ActionType.STORAGE_DISCHARGE,
                volume_mw=0,
                cost_eur=0,
                marginal_cost_eur_per_mwh=float('inf'),
                cvar_reduction_eur=0,
                var_reduction_eur=0,
                residual_after_mw=current_metrics.residual_p50_mw,
                feasible=False,
                operational_constraints=["No storage available"],
                rationale="Storage not available",
                assumptions=[]
            )
        
        storage = state.storage
        constraints = []
        feasible = True
        
        # Check capacity limits
        max_discharge = storage.available_discharge_mw
        if volume_mw > max_discharge:
            constraints.append(f"Volume exceeds available discharge ({max_discharge:.1f} MW)")
            feasible = False
            volume_mw = max_discharge
        
        # Calculate cost (efficiency loss + degradation)
        energy_mwh = volume_mw * 0.25
        cost = (energy_mwh * storage.cost_per_mwh + 
                storage.degradation_cost_per_cycle * (energy_mwh / storage.capacity_mwh))
        
        # Simulate
        adjusted_state = self._adjust_hedge(state, volume_mw)
        sim_results = self.mc_engine.run_full_simulation(adjusted_state)
        new_metrics = RiskMetricsCalculator.calculate_all_metrics(sim_results, adjusted_state)
        
        cvar_reduction = current_metrics.cvar_95_eur - new_metrics.cvar_95_eur
        var_reduction = current_metrics.var_95_eur - new_metrics.var_95_eur
        
        return MitigationAction(
            action_type=ActionType.STORAGE_DISCHARGE,
            volume_mw=volume_mw,
            cost_eur=cost,
            marginal_cost_eur_per_mwh=storage.cost_per_mwh,
            cvar_reduction_eur=cvar_reduction,
            var_reduction_eur=var_reduction,
            residual_after_mw=new_metrics.residual_p50_mw,
            feasible=feasible,
            operational_constraints=constraints,
            rationale=f"Discharge {volume_mw:.1f} MW from storage",
            assumptions=[
                f"SOC: {storage.current_soc:.1%} → {storage.current_soc - energy_mwh/storage.capacity_mwh:.1%}",
                f"Efficiency: {storage.efficiency:.1%}",
                f"Max power: {storage.max_power_mw:.1f} MW"
            ]
        )
    
    def evaluate_curtailment(self,
                            state: PortfolioState,
                            current_metrics: RiskMetrics,
                            source: str,
                            volume_mw: float) -> MitigationAction:
        """Evaluate PV or Wind curtailment"""
        
        action_type = ActionType.CURTAIL_PV if source == "PV" else ActionType.CURTAIL_WIND
        max_available = (state.pv_curtailment_available_mw if source == "PV" 
                        else state.wind_curtailment_available_mw)
        
        constraints = []
        feasible = True
        
        if volume_mw > max_available:
            constraints.append(f"Volume exceeds available {source} generation ({max_available:.1f} MW)")
            feasible = False
            volume_mw = max_available
        
        # Opportunity cost (lost revenue)
        da_price = state.market_prices.day_ahead_mean
        compensation_rate = state.curtailment_compensation_rate
        cost = volume_mw * da_price * (1 - compensation_rate) * 0.25
        
        # Simulate (curtailment reduces generation, increasing net load)
        adjusted_state = self._adjust_hedge(state, -volume_mw)  # Negative adjustment
        sim_results = self.mc_engine.run_full_simulation(adjusted_state)
        new_metrics = RiskMetricsCalculator.calculate_all_metrics(sim_results, adjusted_state)
        
        cvar_reduction = current_metrics.cvar_95_eur - new_metrics.cvar_95_eur
        var_reduction = current_metrics.var_95_eur - new_metrics.var_95_eur
        
        return MitigationAction(
            action_type=action_type,
            volume_mw=volume_mw,
            cost_eur=cost,
            marginal_cost_eur_per_mwh=da_price * (1 - compensation_rate),
            cvar_reduction_eur=cvar_reduction,
            var_reduction_eur=var_reduction,
            residual_after_mw=new_metrics.residual_p50_mw,
            feasible=feasible,
            operational_constraints=constraints,
            rationale=f"Curtail {volume_mw:.1f} MW {source} generation",
            assumptions=[
                f"Compensation rate: {compensation_rate:.1%}",
                f"DA price: {da_price:.2f} €/MWh",
                f"Available {source}: {max_available:.1f} MW"
            ]
        )
    
    def find_optimal_action(self,
                           state: PortfolioState,
                           current_metrics: RiskMetrics) -> List[MitigationAction]:
        """
        Find and rank all mitigation actions
        Returns sorted list (best to worst)
        """
        actions = []
        
        # Baseline
        actions.append(self.evaluate_do_nothing(state, current_metrics))
        
        # Determine required volume based on residual
        if current_metrics.scenario_type in [ScenarioType.SHORTAGE, ScenarioType.EXTREME_SHORTAGE]:
            # Need to reduce shortage
            target_volume = max(current_metrics.residual_p90_mw, 0)
            
            # Intraday buy
            actions.append(self.evaluate_intraday_buy(state, current_metrics, target_volume))
            
            # Flexibility assets
            for asset in state.demand_flexibility:
                volume = min(target_volume, asset.remaining_capacity)
                if volume > 0:
                    actions.append(self.evaluate_flexibility(state, current_metrics, asset, volume))
            
            # Storage discharge
            if state.storage:
                volume = min(target_volume, state.storage.available_discharge_mw)
                if volume > 0:
                    actions.append(self.evaluate_storage_discharge(state, current_metrics, volume))
        
        elif current_metrics.scenario_type in [ScenarioType.SURPLUS, ScenarioType.EXTREME_SURPLUS]:
            # Need to reduce surplus
            target_volume = abs(min(current_metrics.residual_p10_mw, 0))
            
            # Curtailment
            if state.pv_curtailment_available_mw > 0:
                volume = min(target_volume, state.pv_curtailment_available_mw)
                actions.append(self.evaluate_curtailment(state, current_metrics, "PV", volume))
            
            if state.wind_curtailment_available_mw > 0:
                volume = min(target_volume, state.wind_curtailment_available_mw)
                actions.append(self.evaluate_curtailment(state, current_metrics, "WIND", volume))
        
        # Rank by CVaR reduction per € spent
        def ranking_score(action: MitigationAction) -> float:
            if not action.feasible:
                return -1e9
            if action.cost_eur <= 0:
                return 1e9
            return action.cvar_reduction_eur / action.cost_eur
        
        actions.sort(key=ranking_score, reverse=True)
        
        return actions
    
    def _adjust_hedge(self, state: PortfolioState, adjustment_mw: float) -> PortfolioState:
        """Create new state with adjusted hedge position"""
        import copy
        new_state = copy.deepcopy(state)
        new_state.hedge_position_mw += adjustment_mw
        return new_state


# ================================================================
# 5. DECISION ENGINE
# ================================================================

@dataclass
class Decision:
    """Final decision output with full explainability"""
    timestamp: datetime
    
    # Risk state
    risk_state: RiskState
    scenario_type: ScenarioType
    confidence_pct: float
    
    # Risk metrics
    risk_metrics: RiskMetrics
    
    # Recommended action
    primary_action: MitigationAction
    alternative_actions: List[MitigationAction]
    
    # Governance
    manual_override_required: bool
    escalation_required: bool
    breach_details: Optional[str]
    
    # Explainability
    trigger_condition: str
    decision_rationale: str
    tail_scenario_description: str
    assumption_log: List[str]
    model_uncertainty_flags: List[str]


class DecisionEngine:
    """
    SERE Main Decision Engine
    Orchestrates Monte Carlo → Risk Calculation → Mitigation → Decision
    """
    
    def __init__(self, n_scenarios: int = 10000, random_seed: Optional[int] = None):
        self.mc_engine = MonteCarloEngine(n_scenarios, random_seed)
        self.mitigation_optimizer = MitigationOptimizer(self.mc_engine)
        self.decision_history = []
        self.assumption_log = []
    
    def make_decision(self, state: PortfolioState) -> Decision:
        """
        Master decision function
        Execute full workflow: Simulate → Assess Risk → Mitigate → Decide
        """
        
        # Log assumptions
        self._log_assumptions(state)
        
        # 1. Run Monte Carlo simulation
        simulation_results = self.mc_engine.run_full_simulation(state)
        
        # 2. Calculate risk metrics
        risk_metrics = RiskMetricsCalculator.calculate_all_metrics(simulation_results, state)
        
        # 3. Evaluate mitigation actions
        mitigation_actions = self.mitigation_optimizer.find_optimal_action(state, risk_metrics)
        
        # 4. Select primary action based on risk state
        primary_action, alternatives = self._select_action(mitigation_actions, risk_metrics, state)
        
        # 5. Determine governance requirements
        manual_override, escalation, breach_details = self._check_governance(
            risk_metrics, primary_action, state.risk_appetite
        )
        
        # 6. Generate explainability
        trigger, rationale, tail_description = self._generate_explainability(
            risk_metrics, primary_action, simulation_results, state
        )
        
        # 7. Flag model uncertainty
        uncertainty_flags = self._check_model_uncertainty(risk_metrics, state)
        
        # 8. Create decision
        decision = Decision(
            timestamp=state.timestamp,
            risk_state=risk_metrics.risk_state,
            scenario_type=risk_metrics.scenario_type,
            confidence_pct=risk_metrics.confidence_pct,
            risk_metrics=risk_metrics,
            primary_action=primary_action,
            alternative_actions=alternatives[:3],  # Top 3 alternatives
            manual_override_required=manual_override,
            escalation_required=escalation,
            breach_details=breach_details,
            trigger_condition=trigger,
            decision_rationale=rationale,
            tail_scenario_description=tail_description,
            assumption_log=self.assumption_log.copy(),
            model_uncertainty_flags=uncertainty_flags
        )
        
        # Store in history
        self.decision_history.append(decision)
        
        return decision
    
    def _select_action(self,
                      actions: List[MitigationAction],
                      risk_metrics: RiskMetrics,
                      state: PortfolioState) -> Tuple[MitigationAction, List[MitigationAction]]:
        """Select primary action based on risk state"""
        
        # If HOLD state, do nothing
        if risk_metrics.risk_state == RiskState.HOLD:
            do_nothing = [a for a in actions if a.action_type == ActionType.DO_NOTHING][0]
            return do_nothing, [a for a in actions if a != do_nothing]
        
        # Otherwise, select best feasible action
        feasible_actions = [a for a in actions if a.feasible]
        
        if not feasible_actions:
            # No feasible action - escalate
            do_nothing = [a for a in actions if a.action_type == ActionType.DO_NOTHING][0]
            return do_nothing, actions[1:]
        
        # Select action with best CVaR reduction per cost
        primary = feasible_actions[0]
        alternatives = [a for a in actions if a != primary]
        
        return primary, alternatives
    
    def _check_governance(self,
                         risk_metrics: RiskMetrics,
                         primary_action: MitigationAction,
                         risk_appetite: RiskAppetite) -> Tuple[bool, bool, Optional[str]]:
        """Check governance rules and escalation requirements"""
        
        manual_override = False
        escalation = False
        breach_details = None
        
        # Check cost limits
        if primary_action.cost_eur > risk_appetite.manual_override_threshold_eur:
            manual_override = True
            breach_details = f"Action cost (€{primary_action.cost_eur:.0f}) exceeds manual override threshold"
        
        # Check CVaR breach
        if risk_metrics.cvar_95_eur > risk_appetite.cvar_limit_eur:
            escalation = True
            if breach_details:
                breach_details += f"; CVaR (€{risk_metrics.cvar_95_eur:.0f}) exceeds limit (€{risk_appetite.cvar_limit_eur:.0f})"
            else:
                breach_details = f"CVaR (€{risk_metrics.cvar_95_eur:.0f}) exceeds limit (€{risk_appetite.cvar_limit_eur:.0f})"
        
        # Check VaR breach
        if risk_metrics.var_95_eur > risk_appetite.var_limit_eur:
            escalation = True
            if breach_details:
                breach_details += f"; VaR (€{risk_metrics.var_95_eur:.0f}) exceeds limit (€{risk_appetite.var_limit_eur:.0f})"
            else:
                breach_details = f"VaR (€{risk_metrics.var_95_eur:.0f}) exceeds limit (€{risk_appetite.var_limit_eur:.0f})"
        
        # Critical state always requires escalation
        if risk_metrics.risk_state == RiskState.CRITICAL:
            escalation = True
            if not breach_details:
                breach_details = "Critical risk state - immediate escalation required"
        
        return manual_override, escalation, breach_details
    
    def _generate_explainability(self,
                                risk_metrics: RiskMetrics,
                                primary_action: MitigationAction,
                                simulation_results: Dict[str, Any],
                                state: PortfolioState) -> Tuple[str, str, str]:
        """Generate human-readable explanations"""
        
        # Trigger condition
        if risk_metrics.risk_state == RiskState.CRITICAL:
            trigger = f"CRITICAL: CVaR €{risk_metrics.cvar_95_eur:.0f} >> limit €{state.risk_appetite.cvar_limit_eur:.0f}"
        elif risk_metrics.risk_state == RiskState.ACTION:
            trigger = f"ACTION: CVaR €{risk_metrics.cvar_95_eur:.0f} > limit €{state.risk_appetite.cvar_limit_eur:.0f}"
        elif risk_metrics.risk_state == RiskState.WATCH:
            trigger = f"WATCH: CVaR €{risk_metrics.cvar_95_eur:.0f} approaching limit (75%)"
        else:
            trigger = f"HOLD: CVaR €{risk_metrics.cvar_95_eur:.0f} within limits"
        
        # Decision rationale
        if primary_action.action_type == ActionType.DO_NOTHING:
            rationale = f"No action required. Risk position acceptable within governance limits."
        else:
            rationale = (f"Selected {primary_action.action_type.value}: {primary_action.volume_mw:.1f} MW. "
                        f"Reduces CVaR by €{primary_action.cvar_reduction_eur:.0f} at cost €{primary_action.cost_eur:.0f}. "
                        f"Brings residual from {risk_metrics.residual_p50_mw:.1f} MW to {primary_action.residual_after_mw:.1f} MW.")
        
        # Tail scenario description
        tail = risk_metrics.tail_statistics
        tail_description = (
            f"Worst 5% scenarios: {tail['count']} cases with mean loss €{tail['mean_loss']:.0f}, "
            f"max €{tail['max_loss']:.0f}. Dominated by demand={tail['mean_demand']:.1f} MW, "
            f"PV={tail['mean_pv']:.1f} MW, wind={tail['mean_wind']:.1f} MW. "
            f"Residual in tail: {tail['mean_residual']:.1f} MW (max {tail['max_residual']:.1f} MW)."
        )
        
        return trigger, rationale, tail_description
    
    def _check_model_uncertainty(self,
                                risk_metrics: RiskMetrics,
                                state: PortfolioState) -> List[str]:
        """Flag significant model uncertainties"""
        flags = []
        
        # High forecast uncertainty
        cv_demand = state.demand.coefficient_of_variation
        cv_pv = state.pv_generation.coefficient_of_variation
        cv_wind = state.wind_generation.coefficient_of_variation
        
        if cv_demand > 0.3:
            flags.append(f"High demand uncertainty (CV={cv_demand:.1%})")
        if cv_pv > 0.5:
            flags.append(f"High PV uncertainty (CV={cv_pv:.1%})")
        if cv_wind > 0.5:
            flags.append(f"High wind uncertainty (CV={cv_wind:.1%})")
        
        # Low confidence
        if risk_metrics.confidence_pct < 60:
            flags.append(f"Low confidence score ({risk_metrics.confidence_pct:.0f}%)")
        
        # Extreme probabilities
        if risk_metrics.prob_extreme_shortage > 0.3:
            flags.append(f"High probability of extreme shortage ({risk_metrics.prob_extreme_shortage:.1%})")
        
        # Price uncertainty
        if state.market_prices.intraday_bid_std > state.market_prices.intraday_bid_mean * 0.5:
            flags.append("High intraday price volatility")
        
        return flags
    
    def _log_assumptions(self, state: PortfolioState):
        """Log all key assumptions for audit trail"""
        self.assumption_log = [
            f"Timestamp: {state.timestamp}",
            f"Demand: {state.demand.mean:.1f} ± {state.demand.std:.1f} MW",
            f"PV: {state.pv_generation.mean:.1f} ± {state.pv_generation.std:.1f} MW",
            f"Wind: {state.wind_generation.mean:.1f} ± {state.wind_generation.std:.1f} MW",
            f"Correlations: D-PV={state.correlations.demand_pv:.2f}, D-W={state.correlations.demand_wind:.2f}, PV-W={state.correlations.pv_wind:.2f}",
            f"Hedge: {state.hedge_position_mw:.1f} MW",
            f"ReBAP+: {state.market_prices.rebap_plus_mean:.2f} ± {state.market_prices.rebap_plus_std:.2f} €/MWh",
            f"ReBAP-: {state.market_prices.rebap_minus_mean:.2f} ± {state.market_prices.rebap_minus_std:.2f} €/MWh",
            f"Intraday bid: {state.market_prices.intraday_bid_mean:.2f} ± {state.market_prices.intraday_bid_std:.2f} €/MWh",
            f"Risk appetite: CVaR limit €{state.risk_appetite.cvar_limit_eur:.0f}, VaR limit €{state.risk_appetite.var_limit_eur:.0f}",
            f"Confidence level: {state.risk_appetite.confidence_level:.1%}"
        ]


# ================================================================
# 6. CONVENIENCE FUNCTIONS
# ================================================================

def create_example_state() -> PortfolioState:
    """Create example portfolio state for testing"""
    
    demand = UncertaintyInput(
        name="Demand",
        mean=100.0,
        std=15.0,
        p10=85.0,
        p50=100.0,
        p90=115.0
    )
    
    pv = UncertaintyInput(
        name="PV",
        mean=20.0,
        std=8.0,
        p10=12.0,
        p50=20.0,
        p90=28.0
    )
    
    wind = UncertaintyInput(
        name="Wind",
        mean=30.0,
        std=12.0,
        p10=18.0,
        p50=30.0,
        p90=42.0
    )
    
    correlations = CorrelationMatrix(
        demand_pv=-0.3,
        demand_wind=0.1,
        pv_wind=-0.2
    )
    
    flex_asset = FlexibilityAsset(
        name="Digital Battery",
        max_mw=10.0,
        duration_minutes=60,
        cost_per_mwh=80.0
    )
    
    storage = StorageAsset(
        capacity_mwh=20.0,
        max_power_mw=10.0,
        efficiency=0.90,
        cost_per_mwh=50.0
    )
    
    prices = MarketPrices(
        day_ahead_mean=100.0,
        day_ahead_std=20.0,
        intraday_bid_mean=150.0,
        intraday_bid_std=30.0,
        intraday_ask_mean=140.0,
        intraday_ask_std=30.0,
        rebap_plus_mean=300.0,
        rebap_plus_std=100.0,
        rebap_plus_p95=450.0,
        rebap_minus_mean=-50.0,
        rebap_minus_std=20.0,
        rebap_minus_p95=-80.0
    )
    
    risk_appetite = RiskAppetite(
        cvar_limit_eur=1000.0,
        var_limit_eur=800.0,
        confidence_level=0.95,
        lambda_risk_aversion=2.0
    )
    
    return PortfolioState(
        timestamp=datetime.now(),
        demand=demand,
        pv_generation=pv,
        wind_generation=wind,
        correlations=correlations,
        hedge_position_mw=50.0,
        demand_flexibility=[flex_asset],
        storage=storage,
        pv_curtailment_available_mw=15.0,
        wind_curtailment_available_mw=25.0,
        curtailment_compensation_rate=0.0,
        market_prices=prices,
        risk_appetite=risk_appetite
    )


if __name__ == "__main__":
    """Test SERE engine"""
    print("=" * 80)
    print("SERE - Sustainable Exergy Risk Engine")
    print("Test Suite")
    print("=" * 80)
    
    # Create example state
    state = create_example_state()
    
    # Initialize engine
    engine = DecisionEngine(n_scenarios=10000, random_seed=42)
    
    # Make decision
    print("\nRunning Monte Carlo simulation with 10,000 scenarios...")
    decision = engine.make_decision(state)
    
    # Display results
    print("\n" + "=" * 80)
    print("DECISION OUTPUT")
    print("=" * 80)
    
    print(f"\nTimestamp: {decision.timestamp}")
    print(f"Risk State: {decision.risk_state.value}")
    print(f"Scenario Type: {decision.scenario_type.value}")
    print(f"Confidence: {decision.confidence_pct:.1f}%")
    
    print(f"\n--- RISK METRICS ---")
    rm = decision.risk_metrics
    print(f"Expected Loss: €{rm.expected_loss_eur:.2f}")
    print(f"VaR (95%): €{rm.var_95_eur:.2f}")
    print(f"CVaR (95%): €{rm.cvar_95_eur:.2f}")
    print(f"Risk Energy: €{rm.risk_energy_eur:.2f}")
    print(f"Residual: P10={rm.residual_p10_mw:.1f}, P50={rm.residual_p50_mw:.1f}, P90={rm.residual_p90_mw:.1f} MW")
    print(f"Prob(Shortage): {rm.prob_shortage:.1%}, Prob(Surplus): {rm.prob_surplus:.1%}")
    
    print(f"\n--- RECOMMENDED ACTION ---")
    pa = decision.primary_action
    print(f"Action: {pa.action_type.value}")
    print(f"Volume: {pa.volume_mw:.1f} MW")
    print(f"Cost: €{pa.cost_eur:.2f}")
    print(f"CVaR Reduction: €{pa.cvar_reduction_eur:.2f}")
    print(f"Feasible: {pa.feasible}")
    print(f"Rationale: {pa.rationale}")
    
    print(f"\n--- EXPLAINABILITY ---")
    print(f"Trigger: {decision.trigger_condition}")
    print(f"Rationale: {decision.decision_rationale}")
    print(f"Tail Scenarios: {decision.tail_scenario_description}")
    
    if decision.model_uncertainty_flags:
        print(f"\n--- MODEL UNCERTAINTY FLAGS ---")
        for flag in decision.model_uncertainty_flags:
            print(f"  • {flag}")
    
    if decision.breach_details:
        print(f"\n--- GOVERNANCE BREACH ---")
        print(f"Manual Override Required: {decision.manual_override_required}")
        print(f"Escalation Required: {decision.escalation_required}")
        print(f"Details: {decision.breach_details}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

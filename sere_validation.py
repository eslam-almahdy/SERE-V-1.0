"""
SERE Validation & Backtesting Module
Supports model validation, stress testing, and performance monitoring
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from sere_core import *
from sere_database import SEREDatabase


class StressTester:
    """
    Stress testing framework for SERE
    Tests extreme but plausible scenarios
    """
    
    def __init__(self, engine: DecisionEngine):
        self.engine = engine
        self.stress_scenarios = []
    
    def add_stress_scenario(self, name: str, state: PortfolioState):
        """Add a stress test scenario"""
        self.stress_scenarios.append({
            'name': name,
            'state': state
        })
    
    def run_all_stress_tests(self) -> pd.DataFrame:
        """Execute all stress tests"""
        
        results = []
        
        for scenario in self.stress_scenarios:
            decision = self.engine.make_decision(scenario['state'])
            
            results.append({
                'scenario_name': scenario['name'],
                'risk_state': decision.risk_state.value,
                'cvar_95_eur': decision.risk_metrics.cvar_95_eur,
                'var_95_eur': decision.risk_metrics.var_95_eur,
                'expected_loss_eur': decision.risk_metrics.expected_loss_eur,
                'residual_p90_mw': decision.risk_metrics.residual_p90_mw,
                'primary_action': decision.primary_action.action_type.value,
                'action_cost_eur': decision.primary_action.cost_eur,
                'escalation_required': decision.escalation_required
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def create_extreme_shortage_scenario(base_state: PortfolioState) -> PortfolioState:
        """Create extreme shortage stress scenario"""
        import copy
        stress_state = copy.deepcopy(base_state)
        
        # Increase demand uncertainty
        stress_state.demand.mean *= 1.3
        stress_state.demand.std *= 2.0
        stress_state.demand.p90 = stress_state.demand.mean + 2 * stress_state.demand.std
        
        # Reduce generation
        stress_state.pv_generation.mean *= 0.3
        stress_state.wind_generation.mean *= 0.3
        
        # Increase ReBAP penalties
        stress_state.market_prices.rebap_plus_mean *= 2.0
        stress_state.market_prices.rebap_plus_p95 *= 2.5
        
        return stress_state
    
    @staticmethod
    def create_extreme_surplus_scenario(base_state: PortfolioState) -> PortfolioState:
        """Create extreme surplus stress scenario"""
        import copy
        stress_state = copy.deepcopy(base_state)
        
        # Decrease demand
        stress_state.demand.mean *= 0.6
        
        # Increase generation
        stress_state.pv_generation.mean *= 1.8
        stress_state.wind_generation.mean *= 1.8
        
        # Increase ReBAP penalties
        stress_state.market_prices.rebap_minus_mean *= 2.0
        
        return stress_state
    
    @staticmethod
    def create_high_volatility_scenario(base_state: PortfolioState) -> PortfolioState:
        """Create high volatility stress scenario"""
        import copy
        stress_state = copy.deepcopy(base_state)
        
        # Increase all uncertainties
        stress_state.demand.std *= 3.0
        stress_state.pv_generation.std *= 3.0
        stress_state.wind_generation.std *= 3.0
        
        # Increase price volatility
        stress_state.market_prices.intraday_bid_std *= 2.5
        stress_state.market_prices.rebap_plus_std *= 2.5
        
        return stress_state
    
    @staticmethod
    def create_low_liquidity_scenario(base_state: PortfolioState) -> PortfolioState:
        """Create low market liquidity stress scenario"""
        import copy
        stress_state = copy.deepcopy(base_state)
        
        # Reduce liquidity
        stress_state.market_prices.liquidity_indicator = 0.2
        
        # Widen bid-ask spread
        stress_state.market_prices.intraday_bid_mean *= 1.5
        stress_state.market_prices.intraday_ask_mean *= 0.7
        
        return stress_state


class Backtester:
    """
    Backtesting engine for SERE
    Compares forecasts vs actuals and measures model accuracy
    """
    
    def __init__(self, engine: DecisionEngine, database: SEREDatabase):
        self.engine = engine
        self.database = database
    
    def run_backtest(self, 
                     forecast_state: PortfolioState,
                     actual_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Run single backtest
        
        actual_data should contain:
        - actual_demand
        - actual_pv
        - actual_wind
        - actual_cost (realized cost)
        """
        
        # Run decision on forecast
        decision = self.engine.make_decision(forecast_state)
        
        # Calculate forecast values
        forecast_demand = forecast_state.demand.p50
        forecast_pv = forecast_state.pv_generation.p50
        forecast_wind = forecast_state.wind_generation.p50
        forecast_net_load = forecast_demand - forecast_pv - forecast_wind
        forecast_residual = forecast_net_load - forecast_state.hedge_position_mw
        
        # Calculate actual values
        actual_demand = actual_data['actual_demand']
        actual_pv = actual_data['actual_pv']
        actual_wind = actual_data['actual_wind']
        actual_net_load = actual_demand - actual_pv - actual_wind
        actual_residual = actual_net_load - forecast_state.hedge_position_mw
        
        # Calculate errors
        demand_error = actual_demand - forecast_demand
        pv_error = actual_pv - forecast_pv
        wind_error = actual_wind - forecast_wind
        residual_error = actual_residual - forecast_residual
        
        # Calculate cost error
        forecast_cvar = decision.risk_metrics.cvar_95_eur
        actual_cost = actual_data.get('actual_cost', 0)
        cost_error = actual_cost - forecast_cvar
        
        # Assess forecast quality
        demand_mape = abs(demand_error) / forecast_demand * 100 if forecast_demand > 0 else 0
        
        # Check if actual fell within forecast range
        demand_within_range = (forecast_state.demand.p10 <= actual_demand <= forecast_state.demand.p90)
        pv_within_range = (forecast_state.pv_generation.p10 <= actual_pv <= forecast_state.pv_generation.p90)
        wind_within_range = (forecast_state.wind_generation.p10 <= actual_wind <= forecast_state.wind_generation.p90)
        
        # Store in database
        self.database.store_backtesting_result(
            forecast_data={
                'demand_p50': forecast_demand,
                'pv_p50': forecast_pv,
                'wind_p50': forecast_wind,
                'residual_p50': forecast_residual,
                'cvar_eur': forecast_cvar
            },
            actual_data=actual_data,
            decision=decision
        )
        
        return {
            'forecast': {
                'demand': forecast_demand,
                'pv': forecast_pv,
                'wind': forecast_wind,
                'residual': forecast_residual,
                'cvar': forecast_cvar
            },
            'actual': {
                'demand': actual_demand,
                'pv': actual_pv,
                'wind': actual_wind,
                'residual': actual_residual,
                'cost': actual_cost
            },
            'errors': {
                'demand': demand_error,
                'pv': pv_error,
                'wind': wind_error,
                'residual': residual_error,
                'cost': cost_error
            },
            'metrics': {
                'demand_mape': demand_mape,
                'demand_within_range': demand_within_range,
                'pv_within_range': pv_within_range,
                'wind_within_range': wind_within_range
            },
            'decision': decision
        }
    
    def calculate_forecast_accuracy(self, backtest_results: List[Dict]) -> Dict[str, float]:
        """Calculate overall forecast accuracy metrics"""
        
        if not backtest_results:
            return {}
        
        demand_errors = [r['errors']['demand'] for r in backtest_results]
        residual_errors = [r['errors']['residual'] for r in backtest_results]
        cost_errors = [r['errors']['cost'] for r in backtest_results]
        
        return {
            'demand_mae': np.mean(np.abs(demand_errors)),
            'demand_rmse': np.sqrt(np.mean(np.square(demand_errors))),
            'demand_bias': np.mean(demand_errors),
            'residual_mae': np.mean(np.abs(residual_errors)),
            'residual_rmse': np.sqrt(np.mean(np.square(residual_errors))),
            'cost_mae': np.mean(np.abs(cost_errors)),
            'cost_bias': np.mean(cost_errors),
            'n_tests': len(backtest_results)
        }


class ModelValidator:
    """
    Model validation framework
    Checks model assumptions and flags issues
    """
    
    @staticmethod
    def validate_uncertainty_inputs(state: PortfolioState) -> List[str]:
        """Validate uncertainty inputs for consistency"""
        
        warnings = []
        
        # Check demand
        if state.demand.std <= 0:
            warnings.append("Demand std deviation must be positive")
        
        if not (state.demand.p10 <= state.demand.p50 <= state.demand.p90):
            warnings.append("Demand percentiles not properly ordered")
        
        cv_demand = state.demand.coefficient_of_variation
        if cv_demand > 0.5:
            warnings.append(f"Very high demand uncertainty (CV={cv_demand:.1%})")
        
        # Check PV
        if state.pv_generation.p10 < 0:
            warnings.append("PV P10 cannot be negative")
        
        if not (state.pv_generation.p10 <= state.pv_generation.p50 <= state.pv_generation.p90):
            warnings.append("PV percentiles not properly ordered")
        
        # Check Wind
        if state.wind_generation.p10 < 0:
            warnings.append("Wind P10 cannot be negative")
        
        if not (state.wind_generation.p10 <= state.wind_generation.p50 <= state.wind_generation.p90):
            warnings.append("Wind percentiles not properly ordered")
        
        # Check correlations
        corr_matrix = state.correlations.get_matrix()
        eigenvalues = np.linalg.eigvals(corr_matrix)
        if np.any(eigenvalues <= 0):
            warnings.append("Correlation matrix is not positive definite")
        
        return warnings
    
    @staticmethod
    def validate_market_prices(state: PortfolioState) -> List[str]:
        """Validate market price assumptions"""
        
        warnings = []
        
        prices = state.market_prices
        
        # Check bid-ask spread
        if prices.intraday_bid_mean <= prices.intraday_ask_mean:
            warnings.append("Intraday bid should be > ask (buy > sell)")
        
        # Check ReBAP penalties are higher than intraday
        if prices.rebap_plus_mean < prices.intraday_bid_mean:
            warnings.append("ReBAP+ should be > intraday bid (penalty > market)")
        
        # Check for negative prices
        if prices.day_ahead_mean < 0 and not state.negative_price_compensation:
            warnings.append("Negative prices detected but compensation disabled")
        
        # Check volatility
        if prices.intraday_bid_std / prices.intraday_bid_mean > 1.0:
            warnings.append("Extremely high intraday price volatility (>100%)")
        
        return warnings
    
    @staticmethod
    def validate_risk_appetite(state: PortfolioState) -> List[str]:
        """Validate risk appetite settings"""
        
        warnings = []
        
        ra = state.risk_appetite
        
        # Check limits are positive
        if ra.cvar_limit_eur <= 0:
            warnings.append("CVaR limit must be positive")
        
        if ra.var_limit_eur <= 0:
            warnings.append("VaR limit must be positive")
        
        # Check consistency
        if ra.cvar_limit_eur < ra.var_limit_eur:
            warnings.append("CVaR limit should be >= VaR limit (CVaR > VaR)")
        
        # Check confidence level
        if ra.confidence_level < 0.9 or ra.confidence_level > 0.99:
            warnings.append(f"Unusual confidence level: {ra.confidence_level:.1%}")
        
        # Check thresholds
        if ra.action_threshold_mw <= ra.watch_threshold_mw:
            warnings.append("ACTION threshold should be > WATCH threshold")
        
        if ra.watch_threshold_mw <= ra.hold_threshold_mw:
            warnings.append("WATCH threshold should be > HOLD threshold")
        
        return warnings
    
    @staticmethod
    def validate_full_state(state: PortfolioState) -> Dict[str, List[str]]:
        """Run all validation checks"""
        
        return {
            'uncertainty_warnings': ModelValidator.validate_uncertainty_inputs(state),
            'price_warnings': ModelValidator.validate_market_prices(state),
            'risk_appetite_warnings': ModelValidator.validate_risk_appetite(state)
        }


class GovernanceMonitor:
    """
    Governance monitoring and compliance tracking
    """
    
    def __init__(self, database: SEREDatabase):
        self.database = database
    
    def check_cvar_compliance(self, days: int = 30) -> Dict[str, Any]:
        """Check CVaR limit compliance"""
        return self.database.get_cvar_compliance_rate(days)
    
    def check_escalation_rate(self, days: int = 30) -> Dict[str, float]:
        """Calculate escalation rate"""
        
        history = self.database.get_decision_history(days)
        
        if len(history) == 0:
            return {'escalation_rate': 0, 'total': 0, 'escalations': 0}
        
        escalations = history['escalation_required'].sum()
        total = len(history)
        
        return {
            'escalation_rate': (escalations / total * 100) if total > 0 else 0,
            'total': total,
            'escalations': int(escalations)
        }
    
    def check_manual_override_rate(self, days: int = 30) -> Dict[str, float]:
        """Calculate manual override rate"""
        
        history = self.database.get_decision_history(days)
        
        if len(history) == 0:
            return {'override_rate': 0, 'total': 0, 'overrides': 0}
        
        overrides = history['manual_override_required'].sum()
        total = len(history)
        
        return {
            'override_rate': (overrides / total * 100) if total > 0 else 0,
            'total': total,
            'overrides': int(overrides)
        }
    
    def generate_governance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive governance report"""
        
        cvar_compliance = self.check_cvar_compliance(days)
        escalation_stats = self.check_escalation_rate(days)
        override_stats = self.check_manual_override_rate(days)
        action_dist = self.database.get_action_distribution(days)
        
        return {
            'period_days': days,
            'cvar_compliance': cvar_compliance,
            'escalation_statistics': escalation_stats,
            'manual_override_statistics': override_stats,
            'action_distribution': action_dist.to_dict('records') if not action_dist.empty else [],
            'generated_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    """Test validation module"""
    
    print("=" * 80)
    print("SERE Validation Module - Test Suite")
    print("=" * 80)
    
    # Create test state
    from sere_core import create_example_state, DecisionEngine
    from sere_database import SEREDatabase
    
    state = create_example_state()
    engine = DecisionEngine(n_scenarios=1000, random_seed=42)
    db = SEREDatabase("sere_test.db")
    
    # Test model validation
    print("\n--- MODEL VALIDATION ---")
    validator = ModelValidator()
    validation_results = validator.validate_full_state(state)
    
    all_clear = True
    for category, warnings in validation_results.items():
        if warnings:
            all_clear = False
            print(f"\n{category}:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
    
    if all_clear:
        print("✅ All validation checks passed")
    
    # Test stress testing
    print("\n--- STRESS TESTING ---")
    stress_tester = StressTester(engine)
    
    stress_tester.add_stress_scenario("Extreme Shortage", 
                                     StressTester.create_extreme_shortage_scenario(state))
    stress_tester.add_stress_scenario("Extreme Surplus", 
                                     StressTester.create_extreme_surplus_scenario(state))
    stress_tester.add_stress_scenario("High Volatility", 
                                     StressTester.create_high_volatility_scenario(state))
    stress_tester.add_stress_scenario("Low Liquidity", 
                                     StressTester.create_low_liquidity_scenario(state))
    
    stress_results = stress_tester.run_all_stress_tests()
    print("\n✅ Stress test results:")
    print(stress_results[['scenario_name', 'risk_state', 'cvar_95_eur', 'primary_action']])
    
    # Test backtesting
    print("\n--- BACKTESTING ---")
    backtester = Backtester(engine, db)
    
    actual_data = {
        'actual_demand': 105.0,
        'actual_pv': 18.0,
        'actual_wind': 28.0,
        'actual_cost': 750.0
    }
    
    backtest_result = backtester.run_backtest(state, actual_data)
    print("✅ Backtest completed")
    print(f"   Demand error: {backtest_result['errors']['demand']:.1f} MW")
    print(f"   Residual error: {backtest_result['errors']['residual']:.1f} MW")
    print(f"   Cost error: €{backtest_result['errors']['cost']:.0f}")
    
    # Test governance monitoring
    print("\n--- GOVERNANCE MONITORING ---")
    gov_monitor = GovernanceMonitor(db)
    
    report = gov_monitor.generate_governance_report(days=1)
    print(f"✅ Governance report generated")
    print(f"   CVaR Compliance: {report['cvar_compliance']['compliance_rate']:.1f}%")
    print(f"   Escalation Rate: {report['escalation_statistics']['escalation_rate']:.1f}%")
    
    print("\n" + "=" * 80)
    print("Validation module test completed successfully!")
    print("=" * 80)

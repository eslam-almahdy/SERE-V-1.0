"""
SERE Database Module
Stores scenarios, risk metrics, decisions, and backtesting data
Supports audit trail and historical analysis
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from sere_core import Decision, RiskMetrics, MitigationAction, PortfolioState


class SEREDatabase:
    """
    Database manager for SERE system
    Stores all decisions, risk metrics, and scenarios for audit and backtesting
    """
    
    def __init__(self, db_path: str = "sere_data.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                risk_state TEXT NOT NULL,
                scenario_type TEXT NOT NULL,
                confidence_pct REAL,
                cvar_95_eur REAL,
                var_95_eur REAL,
                expected_loss_eur REAL,
                risk_energy_eur REAL,
                residual_p10_mw REAL,
                residual_p50_mw REAL,
                residual_p90_mw REAL,
                primary_action_type TEXT,
                primary_action_volume_mw REAL,
                primary_action_cost_eur REAL,
                manual_override_required INTEGER,
                escalation_required INTEGER,
                breach_details TEXT,
                trigger_condition TEXT,
                decision_rationale TEXT,
                tail_scenario_description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk metrics table (historical time series)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                n_scenarios INTEGER,
                expected_loss_eur REAL,
                var_95_eur REAL,
                cvar_95_eur REAL,
                risk_energy_eur REAL,
                expected_net_load_mw REAL,
                residual_p10_mw REAL,
                residual_p50_mw REAL,
                residual_p90_mw REAL,
                short_exposure_mw REAL,
                surplus_exposure_mw REAL,
                prob_shortage REAL,
                prob_surplus REAL,
                prob_extreme_shortage REAL,
                confidence_pct REAL,
                risk_state TEXT,
                scenario_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Mitigation actions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mitigation_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER,
                action_type TEXT NOT NULL,
                volume_mw REAL,
                cost_eur REAL,
                marginal_cost_eur_per_mwh REAL,
                cvar_reduction_eur REAL,
                var_reduction_eur REAL,
                residual_after_mw REAL,
                feasible INTEGER,
                rationale TEXT,
                rank INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (decision_id) REFERENCES decisions (id)
            )
        """)
        
        # Assumptions log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assumptions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER,
                assumption_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (decision_id) REFERENCES decisions (id)
            )
        """)
        
        # Model uncertainty flags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uncertainty_flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER,
                flag_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (decision_id) REFERENCES decisions (id)
            )
        """)
        
        # Backtesting results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtesting_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_date TEXT NOT NULL,
                decision_timestamp TEXT NOT NULL,
                forecast_demand_p50 REAL,
                actual_demand REAL,
                forecast_pv_p50 REAL,
                actual_pv REAL,
                forecast_wind_p50 REAL,
                actual_wind REAL,
                forecast_residual_p50 REAL,
                actual_residual REAL,
                forecast_cvar_eur REAL,
                actual_cost_eur REAL,
                forecast_error_mw REAL,
                cost_error_eur REAL,
                action_taken TEXT,
                action_effectiveness REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Portfolio states (for replay/audit)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                demand_mean REAL,
                demand_std REAL,
                pv_mean REAL,
                pv_std REAL,
                wind_mean REAL,
                wind_std REAL,
                hedge_position_mw REAL,
                market_prices_json TEXT,
                correlations_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_decision(self, decision: Decision) -> int:
        """
        Store a decision and all related data
        Returns: decision_id
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store main decision
        cursor.execute("""
            INSERT INTO decisions (
                timestamp, risk_state, scenario_type, confidence_pct,
                cvar_95_eur, var_95_eur, expected_loss_eur, risk_energy_eur,
                residual_p10_mw, residual_p50_mw, residual_p90_mw,
                primary_action_type, primary_action_volume_mw, primary_action_cost_eur,
                manual_override_required, escalation_required, breach_details,
                trigger_condition, decision_rationale, tail_scenario_description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.timestamp.isoformat(),
            decision.risk_state.value,
            decision.scenario_type.value,
            decision.confidence_pct,
            decision.risk_metrics.cvar_95_eur,
            decision.risk_metrics.var_95_eur,
            decision.risk_metrics.expected_loss_eur,
            decision.risk_metrics.risk_energy_eur,
            decision.risk_metrics.residual_p10_mw,
            decision.risk_metrics.residual_p50_mw,
            decision.risk_metrics.residual_p90_mw,
            decision.primary_action.action_type.value,
            decision.primary_action.volume_mw,
            decision.primary_action.cost_eur,
            1 if decision.manual_override_required else 0,
            1 if decision.escalation_required else 0,
            decision.breach_details,
            decision.trigger_condition,
            decision.decision_rationale,
            decision.tail_scenario_description
        ))
        
        decision_id = cursor.lastrowid
        
        # Store primary action
        self._store_action(cursor, decision_id, decision.primary_action, 1)
        
        # Store alternative actions
        for rank, action in enumerate(decision.alternative_actions, start=2):
            self._store_action(cursor, decision_id, action, rank)
        
        # Store assumptions
        for assumption in decision.assumption_log:
            cursor.execute("""
                INSERT INTO assumptions_log (decision_id, assumption_text)
                VALUES (?, ?)
            """, (decision_id, assumption))
        
        # Store uncertainty flags
        for flag in decision.model_uncertainty_flags:
            cursor.execute("""
                INSERT INTO uncertainty_flags (decision_id, flag_text)
                VALUES (?, ?)
            """, (decision_id, flag))
        
        conn.commit()
        conn.close()
        
        return decision_id
    
    def _store_action(self, cursor, decision_id: int, action: MitigationAction, rank: int):
        """Store a mitigation action"""
        cursor.execute("""
            INSERT INTO mitigation_actions (
                decision_id, action_type, volume_mw, cost_eur,
                marginal_cost_eur_per_mwh, cvar_reduction_eur,
                var_reduction_eur, residual_after_mw, feasible,
                rationale, rank
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision_id,
            action.action_type.value,
            action.volume_mw,
            action.cost_eur,
            action.marginal_cost_eur_per_mwh,
            action.cvar_reduction_eur,
            action.var_reduction_eur,
            action.residual_after_mw,
            1 if action.feasible else 0,
            action.rationale,
            rank
        ))
    
    def store_risk_metrics(self, risk_metrics: RiskMetrics):
        """Store risk metrics time series"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO risk_metrics (
                timestamp, n_scenarios, expected_loss_eur, var_95_eur,
                cvar_95_eur, risk_energy_eur, expected_net_load_mw,
                residual_p10_mw, residual_p50_mw, residual_p90_mw,
                short_exposure_mw, surplus_exposure_mw,
                prob_shortage, prob_surplus, prob_extreme_shortage,
                confidence_pct, risk_state, scenario_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            risk_metrics.timestamp.isoformat(),
            risk_metrics.n_scenarios,
            risk_metrics.expected_loss_eur,
            risk_metrics.var_95_eur,
            risk_metrics.cvar_95_eur,
            risk_metrics.risk_energy_eur,
            risk_metrics.expected_net_load_mw,
            risk_metrics.residual_p10_mw,
            risk_metrics.residual_p50_mw,
            risk_metrics.residual_p90_mw,
            risk_metrics.short_exposure_mw,
            risk_metrics.surplus_exposure_mw,
            risk_metrics.prob_shortage,
            risk_metrics.prob_surplus,
            risk_metrics.prob_extreme_shortage,
            risk_metrics.confidence_pct,
            risk_metrics.risk_state.value,
            risk_metrics.scenario_type.value
        ))
        
        conn.commit()
        conn.close()
    
    def store_backtesting_result(self, forecast_data: Dict[str, float], 
                                 actual_data: Dict[str, float],
                                 decision: Decision):
        """Store backtesting comparison"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        forecast_error = abs(forecast_data.get('residual_p50', 0) - actual_data.get('actual_residual', 0))
        cost_error = abs(forecast_data.get('cvar_eur', 0) - actual_data.get('actual_cost', 0))
        
        cursor.execute("""
            INSERT INTO backtesting_results (
                test_date, decision_timestamp,
                forecast_demand_p50, actual_demand,
                forecast_pv_p50, actual_pv,
                forecast_wind_p50, actual_wind,
                forecast_residual_p50, actual_residual,
                forecast_cvar_eur, actual_cost_eur,
                forecast_error_mw, cost_error_eur,
                action_taken
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            decision.timestamp.isoformat(),
            forecast_data.get('demand_p50'),
            actual_data.get('actual_demand'),
            forecast_data.get('pv_p50'),
            actual_data.get('actual_pv'),
            forecast_data.get('wind_p50'),
            actual_data.get('actual_wind'),
            forecast_data.get('residual_p50'),
            actual_data.get('actual_residual'),
            forecast_data.get('cvar_eur'),
            actual_data.get('actual_cost'),
            forecast_error,
            cost_error,
            decision.primary_action.action_type.value
        ))
        
        conn.commit()
        conn.close()
    
    def get_decision_history(self, days: int = 30) -> pd.DataFrame:
        """Get decision history for last N days"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                timestamp, risk_state, scenario_type, cvar_95_eur,
                var_95_eur, expected_loss_eur, residual_p50_mw,
                primary_action_type, primary_action_cost_eur,
                manual_override_required, escalation_required
            FROM decisions
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        """.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_risk_metrics_timeseries(self, days: int = 7) -> pd.DataFrame:
        """Get risk metrics time series"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                timestamp, cvar_95_eur, var_95_eur, expected_loss_eur,
                residual_p50_mw, confidence_pct, risk_state
            FROM risk_metrics
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
            ORDER BY timestamp ASC
        """.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_backtesting_summary(self) -> pd.DataFrame:
        """Get backtesting summary statistics"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                AVG(forecast_error_mw) as avg_forecast_error,
                AVG(cost_error_eur) as avg_cost_error,
                AVG(ABS(forecast_error_mw)) as mae_forecast,
                AVG(ABS(cost_error_eur)) as mae_cost,
                COUNT(*) as total_tests
            FROM backtesting_results
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_cvar_compliance_rate(self, days: int = 30) -> Dict[str, float]:
        """Calculate CVaR compliance rate"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN cvar_95_eur <= 1000 THEN 1 ELSE 0 END) as compliant
            FROM decisions
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
        """.format(days))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] > 0:
            compliance_rate = (row[1] / row[0]) * 100
            return {
                'compliance_rate': compliance_rate,
                'total_decisions': row[0],
                'compliant_decisions': row[1]
            }
        
        return {'compliance_rate': 0, 'total_decisions': 0, 'compliant_decisions': 0}
    
    def get_action_distribution(self, days: int = 30) -> pd.DataFrame:
        """Get distribution of actions taken"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                primary_action_type,
                COUNT(*) as count,
                AVG(primary_action_cost_eur) as avg_cost,
                SUM(primary_action_cost_eur) as total_cost
            FROM decisions
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
            GROUP BY primary_action_type
            ORDER BY count DESC
        """.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def export_audit_trail(self, decision_id: int) -> Dict[str, Any]:
        """Export complete audit trail for a decision"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get decision
        cursor.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,))
        decision_row = cursor.fetchone()
        
        if not decision_row:
            conn.close()
            return {}
        
        # Get actions
        cursor.execute("""
            SELECT * FROM mitigation_actions 
            WHERE decision_id = ? 
            ORDER BY rank
        """, (decision_id,))
        actions = cursor.fetchall()
        
        # Get assumptions
        cursor.execute("""
            SELECT assumption_text FROM assumptions_log 
            WHERE decision_id = ?
        """, (decision_id,))
        assumptions = [row[0] for row in cursor.fetchall()]
        
        # Get uncertainty flags
        cursor.execute("""
            SELECT flag_text FROM uncertainty_flags 
            WHERE decision_id = ?
        """, (decision_id,))
        flags = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'decision_id': decision_id,
            'decision': dict(zip([desc[0] for desc in cursor.description], decision_row)),
            'actions': [dict(zip([desc[0] for desc in cursor.description], action)) for action in actions],
            'assumptions': assumptions,
            'uncertainty_flags': flags
        }


if __name__ == "__main__":
    """Test database functionality"""
    
    print("=" * 80)
    print("SERE Database Module - Test Suite")
    print("=" * 80)
    
    # Initialize database
    db = SEREDatabase("sere_test.db")
    print("\n✅ Database initialized")
    
    # Create sample decision
    from sere_core import create_example_state, DecisionEngine
    
    state = create_example_state()
    engine = DecisionEngine(n_scenarios=1000, random_seed=42)
    decision = engine.make_decision(state)
    
    # Store decision
    decision_id = db.store_decision(decision)
    print(f"✅ Decision stored with ID: {decision_id}")
    
    # Store risk metrics
    db.store_risk_metrics(decision.risk_metrics)
    print("✅ Risk metrics stored")
    
    # Get history
    history = db.get_decision_history(days=1)
    print(f"\n✅ Retrieved {len(history)} decisions from history")
    print(history.head())
    
    # Get compliance rate
    compliance = db.get_cvar_compliance_rate(days=1)
    print(f"\n✅ CVaR Compliance Rate: {compliance['compliance_rate']:.1f}%")
    print(f"   Total Decisions: {compliance['total_decisions']}")
    
    # Export audit trail
    audit = db.export_audit_trail(decision_id)
    print(f"\n✅ Exported audit trail with {len(audit['actions'])} actions")
    
    print("\n" + "=" * 80)
    print("Database test completed successfully!")
    print("=" * 80)

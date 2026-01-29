"""
SERE Dashboard - Sustainable Exergy Risk Engine
Multi-user interface with role-based access
Designed for Forecast Team and Management users

Run: streamlit run sere_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from sere_core import *
import json


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="SERE - Sustainable Exergy Risk Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-card h3 {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .critical {
        color: #e74c3c;
        font-weight: bold;
    }
    .action {
        color: #f39c12;
        font-weight: bold;
    }
    .watch {
        color: #3498db;
        font-weight: bold;
    }
    .hold {
        color: #2ecc71;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================

if 'engine' not in st.session_state:
    st.session_state.engine = DecisionEngine(n_scenarios=10000, random_seed=42)

if 'user_role' not in st.session_state:
    st.session_state.user_role = None

if 'validated_forecast' not in st.session_state:
    st.session_state.validated_forecast = None

if 'decision_history' not in st.session_state:
    st.session_state.decision_history = []

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None


# ================================================================
# LOGIN SCREEN
# ================================================================

def login_screen():
    """User role selection"""
    st.markdown('<div class="main-header"><h1>SERE</h1><h3>Sustainable Exergy Risk Engine</h3></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### User Role Selection")
        
        role = st.selectbox(
            "Select your role",
            ["", "Forecast Team", "Management", "Administrator"],
            index=0
        )
        
        if role and role != "":
            st.markdown(f"**Selected Role:** {role}")
            
            if st.button("Login", type="primary", use_container_width=True):
                if role == "Administrator":
                    st.warning("Administrator role coming soon")
                else:
                    st.session_state.user_role = role
                    st.rerun()


# ================================================================
# FORECAST TEAM INTERFACE
# ================================================================

def forecast_team_interface():
    """Interface for Forecast Team - Submit probabilistic forecasts"""
    
    st.markdown("## Forecast Data Input")
    st.markdown("Submit probabilistic forecasts for demand, generation, and prices")
    
    # Input tabs
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Input", "✅ Validation Results", "📈 Historical Comparison"])
    
    with tab1:
        forecast_input_form()
    
    with tab2:
        validation_results()
    
    with tab3:
        historical_comparison()


def forecast_input_form():
    """Forecast input form"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Demand Forecast (MW)")
        demand_p10 = st.number_input("P10 (Pessimistic)", value=85.0, step=1.0, key="demand_p10")
        demand_p50 = st.number_input("P50 (Expected)", value=100.0, step=1.0, key="demand_p50")
        demand_p90 = st.number_input("P90 (Optimistic)", value=115.0, step=1.0, key="demand_p90")
        
        st.markdown("### PV Generation Forecast (MW)")
        pv_p10 = st.number_input("P10", value=12.0, step=1.0, key="pv_p10")
        pv_p50 = st.number_input("P50", value=20.0, step=1.0, key="pv_p50")
        pv_p90 = st.number_input("P90", value=28.0, step=1.0, key="pv_p90")
        
        st.markdown("### Wind Generation Forecast (MW)")
        wind_p10 = st.number_input("P10", value=18.0, step=1.0, key="wind_p10")
        wind_p50 = st.number_input("P50", value=30.0, step=1.0, key="wind_p50")
        wind_p90 = st.number_input("P90", value=42.0, step=1.0, key="wind_p90")
    
    with col2:
        st.markdown("### Market Prices (€/MWh)")
        da_price = st.number_input("Day-Ahead Price", value=100.0, step=5.0, key="da_price")
        intraday_bid = st.number_input("Intraday Bid (Buy)", value=150.0, step=5.0, key="intraday_bid")
        intraday_ask = st.number_input("Intraday Ask (Sell)", value=140.0, step=5.0, key="intraday_ask")
        
        st.markdown("### ReBAP Penalties (€/MWh)")
        rebap_plus = st.number_input("ReBAP+ (Shortage)", value=300.0, step=10.0, key="rebap_plus")
        rebap_minus = st.number_input("ReBAP- (Surplus)", value=-50.0, step=10.0, key="rebap_minus")
        
        st.markdown("### Current Position")
        hedge_position = st.number_input("Hedge Position (MW)", value=50.0, step=1.0, key="hedge")
        
        st.markdown("### Forecast Quality")
        confidence = st.slider("Confidence Level (%)", 0, 100, 80, key="confidence")
        
        notes = st.text_area("Notes/Comments", key="notes")
    
    # Correlations
    with st.expander("Advanced: Correlation Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            corr_demand_pv = st.slider("Demand-PV Correlation", -1.0, 1.0, -0.3, 0.1, key="corr_dpv")
        with col2:
            corr_demand_wind = st.slider("Demand-Wind Correlation", -1.0, 1.0, 0.1, 0.1, key="corr_dw")
        with col3:
            corr_pv_wind = st.slider("PV-Wind Correlation", -1.0, 1.0, -0.2, 0.1, key="corr_pw")
    
    # Validation button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 VALIDATE FORECAST", type="primary", use_container_width=True):
            # Perform validation
            errors = validate_forecast_data(
                demand_p10, demand_p50, demand_p90,
                pv_p10, pv_p50, pv_p90,
                wind_p10, wind_p50, wind_p90
            )
            
            if errors:
                st.error("Validation Failed")
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                st.success("✅ Validation Passed")
                
                # Store validated forecast
                st.session_state.validated_forecast = {
                    'demand': (demand_p10, demand_p50, demand_p90),
                    'pv': (pv_p10, pv_p50, pv_p90),
                    'wind': (wind_p10, wind_p50, wind_p90),
                    'prices': (da_price, intraday_bid, intraday_ask, rebap_plus, rebap_minus),
                    'hedge': hedge_position,
                    'correlations': (corr_demand_pv, corr_demand_wind, corr_pv_wind),
                    'confidence': confidence,
                    'notes': notes,
                    'timestamp': datetime.now()
                }
                
                st.info("Forecast validated and ready for decision engine. Please notify Management team.")


def validate_forecast_data(d10, d50, d90, pv10, pv50, pv90, w10, w50, w90):
    """Validate forecast data"""
    errors = []
    
    # Check percentile ordering
    if not (d10 <= d50 <= d90):
        errors.append("Demand: P10 ≤ P50 ≤ P90 ordering violated")
    if not (pv10 <= pv50 <= pv90):
        errors.append("PV: P10 ≤ P50 ≤ P90 ordering violated")
    if not (w10 <= w50 <= w90):
        errors.append("Wind: P10 ≤ P50 ≤ P90 ordering violated")
    
    # Check non-negativity for generation
    if pv10 < 0 or pv50 < 0 or pv90 < 0:
        errors.append("PV generation cannot be negative")
    if w10 < 0 or w50 < 0 or w90 < 0:
        errors.append("Wind generation cannot be negative")
    
    # Check reasonable ranges
    if d90 - d10 > d50 * 0.5 and d50 > 0:
        errors.append("Demand uncertainty too high (>50% of mean)")
    if pv90 - pv10 > pv50 * 1.0 and pv50 > 0:
        errors.append("PV uncertainty very high (>100% of mean)")
    
    return errors


def validation_results():
    """Display validation results"""
    
    if st.session_state.validated_forecast is None:
        st.info("No validated forecast available. Please submit forecast data first.")
        return
    
    forecast = st.session_state.validated_forecast
    
    st.success("✅ Forecast Validated Successfully")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Validation Time", forecast['timestamp'].strftime("%H:%M:%S"))
        st.metric("Confidence Level", f"{forecast['confidence']}%")
    
    with col2:
        d10, d50, d90 = forecast['demand']
        st.metric("Demand P50", f"{d50:.1f} MW", delta=f"±{(d90-d10)/2:.1f}")
    
    with col3:
        st.metric("Hedge Position", f"{forecast['hedge']:.1f} MW")
    
    st.markdown("### Quality Score")
    
    # Calculate quality score
    quality_score = calculate_quality_score(forecast)
    
    # Display gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=quality_score,
        title={'text': "Overall Quality Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    if forecast['notes']:
        st.markdown("### Notes")
        st.info(forecast['notes'])


def calculate_quality_score(forecast):
    """Calculate forecast quality score"""
    score = 100
    
    # Penalize high uncertainty
    d10, d50, d90 = forecast['demand']
    if d50 > 0:
        uncertainty_pct = (d90 - d10) / d50
        score -= min(20, uncertainty_pct * 50)
    
    # Bonus for high confidence
    score = score * (forecast['confidence'] / 100)
    
    return max(0, min(100, score))


def historical_comparison():
    """Historical forecast comparison"""
    st.markdown("### Historical Forecast Accuracy")
    
    # Placeholder for historical data
    st.info("Historical comparison feature coming soon. Will show forecast vs. actual performance over time.")
    
    # Mock data for visualization
    dates = pd.date_range(start='2026-01-01', periods=30, freq='D')
    forecast_error = np.random.normal(0, 10, 30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=forecast_error,
        mode='lines+markers',
        name='Forecast Error (MW)',
        line=dict(color='blue')
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Forecast Error Trend (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Error (MW)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ================================================================
# MANAGEMENT INTERFACE
# ================================================================

def management_interface():
    """Interface for Management - Execute decisions and monitor"""
    
    st.markdown("## Risk Control & Decision Execution")
    
    if st.session_state.validated_forecast is None:
        st.warning("⚠️ No validated forecast available from Forecast Team")
        st.info("Waiting for Forecast Team to submit and validate forecast data...")
        return
    
    tabs = st.tabs(["🎯 Decision Engine", "📊 Risk Dashboard", "📈 Loss Distribution", "⚙️ Settings"])
    
    with tabs[0]:
        decision_execution()
    
    with tabs[1]:
        risk_dashboard()
    
    with tabs[2]:
        loss_distribution_analysis()
    
    with tabs[3]:
        settings_panel()


def decision_execution():
    """Decision execution interface"""
    
    st.markdown("### Execute SERE Decision Engine")
    
    # Display forecast summary
    forecast = st.session_state.validated_forecast
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        d10, d50, d90 = forecast['demand']
        st.metric("Demand (P50)", f"{d50:.1f} MW", delta=f"±{(d90-d10)/2:.1f}")
    
    with col2:
        pv10, pv50, pv90 = forecast['pv']
        st.metric("PV (P50)", f"{pv50:.1f} MW", delta=f"±{(pv90-pv10)/2:.1f}")
    
    with col3:
        w10, w50, w90 = forecast['wind']
        st.metric("Wind (P50)", f"{w50:.1f} MW", delta=f"±{(w90-w10)/2:.1f}")
    
    with col4:
        st.metric("Hedge", f"{forecast['hedge']:.1f} MW")
    
    # Execute button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 EXECUTE DECISION ENGINE", type="primary", use_container_width=True):
            with st.spinner("Running Monte Carlo simulation with 10,000 scenarios..."):
                # Build portfolio state from forecast
                state = build_portfolio_state_from_forecast(forecast)
                
                # Execute decision engine
                decision = st.session_state.engine.make_decision(state)
                
                # Store results
                st.session_state.decision_history.append(decision)
                st.session_state.latest_decision = decision
                
                # Store simulation results for visualization
                sim_results = st.session_state.engine.mc_engine.run_full_simulation(state)
                st.session_state.simulation_results = sim_results
                
                st.success("✅ Decision engine executed successfully")
                st.rerun()
    
    # Display latest decision
    if hasattr(st.session_state, 'latest_decision'):
        display_decision_output(st.session_state.latest_decision)


def build_portfolio_state_from_forecast(forecast):
    """Build PortfolioState from validated forecast"""
    
    d10, d50, d90 = forecast['demand']
    pv10, pv50, pv90 = forecast['pv']
    w10, w50, w90 = forecast['wind']
    da_price, intraday_bid, intraday_ask, rebap_plus, rebap_minus = forecast['prices']
    corr_dpv, corr_dw, corr_pw = forecast['correlations']
    
    # Create uncertainty inputs
    demand = UncertaintyInput(
        name="Demand",
        mean=d50,
        std=(d90 - d10) / 2.56,  # Approximate from percentiles
        p10=d10,
        p50=d50,
        p90=d90
    )
    
    pv = UncertaintyInput(
        name="PV",
        mean=pv50,
        std=(pv90 - pv10) / 2.56,
        p10=pv10,
        p50=pv50,
        p90=pv90
    )
    
    wind = UncertaintyInput(
        name="Wind",
        mean=w50,
        std=(w90 - w10) / 2.56,
        p10=w10,
        p50=w50,
        p90=w90
    )
    
    correlations = CorrelationMatrix(
        demand_pv=corr_dpv,
        demand_wind=corr_dw,
        pv_wind=corr_pw
    )
    
    # Flexibility assets
    flex_asset = FlexibilityAsset(
        name="Digital Battery",
        max_mw=10.0,
        duration_minutes=60,
        cost_per_mwh=80.0
    )
    
    # Storage
    storage = StorageAsset(
        capacity_mwh=20.0,
        max_power_mw=10.0,
        efficiency=0.90,
        cost_per_mwh=50.0
    )
    
    # Market prices
    prices = MarketPrices(
        day_ahead_mean=da_price,
        day_ahead_std=da_price * 0.2,
        intraday_bid_mean=intraday_bid,
        intraday_bid_std=intraday_bid * 0.2,
        intraday_ask_mean=intraday_ask,
        intraday_ask_std=intraday_ask * 0.2,
        rebap_plus_mean=rebap_plus,
        rebap_plus_std=rebap_plus * 0.3,
        rebap_plus_p95=rebap_plus * 1.5,
        rebap_minus_mean=rebap_minus,
        rebap_minus_std=abs(rebap_minus) * 0.3,
        rebap_minus_p95=rebap_minus * 1.6
    )
    
    # Risk appetite
    risk_appetite = RiskAppetite(
        cvar_limit_eur=1000.0,
        var_limit_eur=800.0,
        confidence_level=0.95
    )
    
    return PortfolioState(
        timestamp=forecast['timestamp'],
        demand=demand,
        pv_generation=pv,
        wind_generation=wind,
        correlations=correlations,
        hedge_position_mw=forecast['hedge'],
        demand_flexibility=[flex_asset],
        storage=storage,
        pv_curtailment_available_mw=15.0,
        wind_curtailment_available_mw=25.0,
        market_prices=prices,
        risk_appetite=risk_appetite
    )


def display_decision_output(decision: Decision):
    """Display decision output"""
    
    st.markdown("---")
    st.markdown("### Decision Output")
    
    # Risk state banner
    risk_state = decision.risk_state.value
    if risk_state == "CRITICAL":
        st.error(f"🚨 Risk State: **{risk_state}**")
    elif risk_state == "ACTION":
        st.warning(f"⚠️ Risk State: **{risk_state}**")
    elif risk_state == "WATCH":
        st.info(f"👁️ Risk State: **{risk_state}**")
    else:
        st.success(f"✅ Risk State: **{risk_state}**")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    rm = decision.risk_metrics
    
    with col1:
        st.metric("CVaR (95%)", f"€{rm.cvar_95_eur:.0f}", 
                 delta=f"Limit: €{decision.risk_metrics.risk_state}" if rm.cvar_95_eur > 1000 else None,
                 delta_color="inverse")
    
    with col2:
        st.metric("VaR (95%)", f"€{rm.var_95_eur:.0f}")
    
    with col3:
        st.metric("Expected Loss", f"€{rm.expected_loss_eur:.0f}")
    
    with col4:
        st.metric("Confidence", f"{rm.confidence_pct:.0f}%")
    
    # Residual position
    st.markdown("#### Residual Position")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("P10", f"{rm.residual_p10_mw:.1f} MW")
    with col2:
        st.metric("P50", f"{rm.residual_p50_mw:.1f} MW")
    with col3:
        st.metric("P90", f"{rm.residual_p90_mw:.1f} MW")
    
    # Recommended action
    st.markdown("#### Recommended Action")
    
    pa = decision.primary_action
    
    action_color = "green" if pa.feasible else "red"
    st.markdown(f"**Action:** <span style='color:{action_color}; font-size:20px;'>{pa.action_type.value}</span>", 
                unsafe_allow_html=True)
    
    if pa.action_type != ActionType.DO_NOTHING:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Volume", f"{pa.volume_mw:.1f} MW")
        with col2:
            st.metric("Cost", f"€{pa.cost_eur:.0f}")
        with col3:
            st.metric("CVaR Reduction", f"€{pa.cvar_reduction_eur:.0f}")
        with col4:
            st.metric("Marginal Cost", f"€{pa.marginal_cost_eur_per_mwh:.1f}/MWh")
        
        st.info(f"**Rationale:** {pa.rationale}")
        
        if pa.assumptions:
            with st.expander("Assumptions"):
                for assumption in pa.assumptions:
                    st.write(f"• {assumption}")
    
    # Explainability
    st.markdown("#### Decision Rationale")
    
    st.markdown(f"**Trigger:** {decision.trigger_condition}")
    st.markdown(f"**Decision:** {decision.decision_rationale}")
    
    with st.expander("Tail Scenario Analysis"):
        st.write(decision.tail_scenario_description)
    
    # Model uncertainty flags
    if decision.model_uncertainty_flags:
        st.markdown("#### ⚠️ Model Uncertainty Flags")
        for flag in decision.model_uncertainty_flags:
            st.warning(f"• {flag}")
    
    # Governance breach
    if decision.breach_details:
        st.markdown("#### 🚨 Governance Breach")
        st.error(decision.breach_details)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Manual Override Required:** {'YES' if decision.manual_override_required else 'NO'}")
        with col2:
            st.markdown(f"**Escalation Required:** {'YES' if decision.escalation_required else 'NO'}")
    
    # Alternative actions
    if decision.alternative_actions:
        with st.expander("Alternative Actions"):
            for i, alt in enumerate(decision.alternative_actions, 1):
                st.markdown(f"**{i}. {alt.action_type.value}**")
                st.write(f"Volume: {alt.volume_mw:.1f} MW, Cost: €{alt.cost_eur:.0f}, CVaR Reduction: €{alt.cvar_reduction_eur:.0f}")
                st.write(f"Rationale: {alt.rationale}")
                st.markdown("---")


def risk_dashboard():
    """Risk dashboard with visualizations"""
    
    if not hasattr(st.session_state, 'latest_decision'):
        st.info("Execute decision engine first to see risk dashboard")
        return
    
    decision = st.session_state.latest_decision
    rm = decision.risk_metrics
    
    st.markdown("### Risk Metrics Dashboard")
    
    # Risk metrics gauges
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rm.cvar_95_eur,
            title={'text': "CVaR (95%)"},
            delta={'reference': 1000, 'prefix': "€"},
            gauge={
                'axis': {'range': [0, 5000]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 1000], 'color': "lightgreen"},
                    {'range': [1000, 1500], 'color': "yellow"},
                    {'range': [1500, 5000], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1000
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rm.confidence_pct,
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightcoral"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rm.prob_shortage * 100,
            title={'text': "Shortage Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkorange"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual position distribution
    st.markdown("### Residual Position Distribution")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['P10', 'P50', 'P90'],
        y=[rm.residual_p10_mw, rm.residual_p50_mw, rm.residual_p90_mw],
        marker_color=['blue', 'green', 'red'],
        text=[f"{rm.residual_p10_mw:.1f}", f"{rm.residual_p50_mw:.1f}", f"{rm.residual_p90_mw:.1f}"],
        textposition='auto'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Residual Position Percentiles",
        yaxis_title="MW",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def loss_distribution_analysis():
    """Loss distribution visualization"""
    
    if st.session_state.simulation_results is None:
        st.info("Execute decision engine first to see loss distribution")
        return
    
    st.markdown("### Monte Carlo Loss Distribution Analysis")
    
    sim_results = st.session_state.simulation_results
    losses = sim_results['losses']
    residual = sim_results['residual']
    
    # Loss histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=losses,
        nbinsx=50,
        name='Loss Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add VaR and CVaR lines
    var_95 = np.percentile(losses, 95)
    cvar_95 = np.mean(losses[losses >= var_95])
    
    fig.add_vline(x=var_95, line_dash="dash", line_color="orange", 
                 annotation_text=f"VaR (95%): €{var_95:.0f}")
    fig.add_vline(x=cvar_95, line_dash="dash", line_color="red", 
                 annotation_text=f"CVaR (95%): €{cvar_95:.0f}")
    
    fig.update_layout(
        title="Loss Distribution (10,000 scenarios)",
        xaxis_title="Loss (€)",
        yaxis_title="Frequency",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Residual vs Loss scatter
    st.markdown("### Residual Position vs Loss")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=residual,
        y=losses,
        mode='markers',
        marker=dict(
            size=3,
            color=losses,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Loss (€)")
        ),
        name='Scenarios'
    ))
    
    fig.update_layout(
        title="Residual Position vs Financial Loss",
        xaxis_title="Residual Position (MW)",
        yaxis_title="Loss (€)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    st.markdown("### Distribution Statistics")
    
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'P5', 'P10', 'P50', 'P90', 'P95', 'Max', 'VaR (95%)', 'CVaR (95%)'],
        'Loss (€)': [
            f"{np.mean(losses):.2f}",
            f"{np.std(losses):.2f}",
            f"{np.min(losses):.2f}",
            f"{np.percentile(losses, 5):.2f}",
            f"{np.percentile(losses, 10):.2f}",
            f"{np.percentile(losses, 50):.2f}",
            f"{np.percentile(losses, 90):.2f}",
            f"{np.percentile(losses, 95):.2f}",
            f"{np.max(losses):.2f}",
            f"{var_95:.2f}",
            f"{cvar_95:.2f}"
        ],
        'Residual (MW)': [
            f"{np.mean(residual):.2f}",
            f"{np.std(residual):.2f}",
            f"{np.min(residual):.2f}",
            f"{np.percentile(residual, 5):.2f}",
            f"{np.percentile(residual, 10):.2f}",
            f"{np.percentile(residual, 50):.2f}",
            f"{np.percentile(residual, 90):.2f}",
            f"{np.percentile(residual, 95):.2f}",
            f"{np.max(residual):.2f}",
            f"{np.percentile(residual, 95):.2f}",
            f"{np.mean(residual[residual >= np.percentile(residual, 95)]):.2f}"
        ]
    })
    
    st.dataframe(stats_df, use_container_width=True)


def settings_panel():
    """Settings and configuration"""
    
    st.markdown("### Risk Appetite Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cvar_limit = st.number_input("CVaR Limit (€)", value=1000.0, step=100.0)
        var_limit = st.number_input("VaR Limit (€)", value=800.0, step=100.0)
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
    with col2:
        lambda_risk = st.slider("Risk Aversion (λ)", 0.0, 5.0, 2.0, 0.1)
        n_scenarios = st.number_input("Monte Carlo Scenarios", value=10000, step=1000, min_value=1000)
    
    st.markdown("### Decision Thresholds (MW)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hold_threshold = st.number_input("HOLD Threshold", value=2.0, step=0.5)
    with col2:
        watch_threshold = st.number_input("WATCH Threshold", value=5.0, step=0.5)
    with col3:
        action_threshold = st.number_input("ACTION Threshold", value=10.0, step=1.0)
    
    if st.button("Apply Settings", type="primary"):
        # Update engine settings
        st.session_state.engine = DecisionEngine(n_scenarios=int(n_scenarios))
        st.success("Settings updated successfully")


# ================================================================
# MAIN APP LOGIC
# ================================================================

def main():
    """Main application logic"""
    
    # Check if user is logged in
    if st.session_state.user_role is None:
        login_screen()
        return
    
    # Header with logout
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown('<div class="main-header"><h1>SERE</h1><h3>Sustainable Exergy Risk Engine</h3></div>', 
                    unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Role:** {st.session_state.user_role}")
        if st.button("Logout"):
            st.session_state.user_role = None
            st.session_state.validated_forecast = None
            st.rerun()
    
    # Route to appropriate interface
    if st.session_state.user_role == "Forecast Team":
        forecast_team_interface()
    elif st.session_state.user_role == "Management":
        management_interface()
    elif st.session_state.user_role == "Administrator":
        st.info("Administrator interface coming soon")


if __name__ == "__main__":
    main()

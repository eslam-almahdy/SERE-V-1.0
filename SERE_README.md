# SERE - Sustainable Exergy Risk Engine

**Advanced Risk Control System with Monte Carlo, VaR & CVaR**

Version 2.0 - Production Ready

---

## 🎯 Overview

SERE (Sustainable Exergy Risk Engine) is a next-generation risk control system for energy portfolio management. Built on advanced Monte Carlo simulation, Value-at-Risk (VaR), and Conditional Value-at-Risk (CVaR) methodologies, SERE provides resilience-focused decision-making under uncertainty.

### Core Philosophy

- **No point forecasts** - All decisions based on probability distributions
- **Tail risk drives decisions** - Rare but extreme events explicitly managed
- **CVaR is primary control** - Expected shortfall, not expected value
- **Resilience over profit** - Capital protection and controlled growth

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    SERE DECISION ENGINE                        │
└────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ MONTE CARLO      │  │ RISK CALCULATOR  │  │ MITIGATION       │
│ SIMULATION       │  │                  │  │ OPTIMIZER        │
│                  │  │ • VaR (95%)      │  │                  │
│ • 10,000+        │  │ • CVaR (95%)     │  │ • Intraday       │
│   scenarios      │  │ • Tail analysis  │  │ • Flexibility    │
│ • Correlations   │  │ • Risk Energy    │  │ • Storage        │
│ • Distributions  │  │ • Probabilities  │  │ • Curtailment    │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    DECISION        │
                    │                    │
                    │ • Risk State       │
                    │ • Primary Action   │
                    │ • Alternatives     │
                    │ • Explainability   │
                    │ • Governance       │
                    └────────────────────┘
```

---

## 📦 Components

### 1. **sere_core.py** (1,400+ lines)
Core decision engine with:
- Monte Carlo simulation engine (≥10,000 scenarios)
- Correlated uncertainty modeling
- VaR and CVaR calculation
- Tail scenario identification
- Mitigation action optimizer
- Full explainability framework

### 2. **sere_dashboard.py** (1,000+ lines)
Streamlit web interface with:
- Role-based access control (Forecast Team, Management)
- Real-time risk visualization
- Loss distribution analysis
- Interactive decision execution
- Governance monitoring

### 3. **sere_database.py** (400+ lines)
SQLite database for:
- Decision history and audit trail
- Risk metrics time series
- Backtesting results
- Assumption logging
- Compliance tracking

### 4. **sere_validation.py** (500+ lines)
Validation and backtesting with:
- Model validation framework
- Stress testing (4 scenarios)
- Backtesting engine
- Governance monitoring
- Forecast accuracy metrics

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download SERE files
cd sere_project

# Install dependencies
pip install -r requirements.txt
```

### Running SERE

```bash
# Test core engine
python sere_core.py

# Launch dashboard
streamlit run sere_dashboard.py

# Test validation module
python sere_validation.py

# Test database
python sere_database.py
```

### First Decision

```python
from sere_core import *

# Create portfolio state
state = create_example_state()

# Initialize engine
engine = DecisionEngine(n_scenarios=10000)

# Make decision
decision = engine.make_decision(state)

# View results
print(f"Risk State: {decision.risk_state.value}")
print(f"CVaR (95%): €{decision.risk_metrics.cvar_95_eur:.0f}")
print(f"Action: {decision.primary_action.action_type.value}")
```

---

## 📊 Key Features

### Monte Carlo Simulation
- **10,000+ scenarios** for robust tail risk assessment
- **Correlated uncertainties** (demand, PV, wind)
- **Multivariate normal distribution** sampling
- **Price scenario generation**

### Risk Metrics
- **VaR (95%)**: Value at Risk - 95th percentile loss
- **CVaR (95%)**: Conditional VaR - expected loss in worst 5%
- **Expected Shortfall**: Tail exposure magnitude
- **Risk Energy**: Expected cost + λ·CVaR composite metric

### Decision Logic
- **Risk States**: HOLD, WATCH, ACTION, CRITICAL
- **Mitigation Actions**: Intraday trading, flexibility, storage, curtailment
- **Cost Optimization**: Minimize CVaR per € spent
- **Governance**: Manual override and escalation triggers

### Explainability
- **Trigger conditions**: Why action was/wasn't taken
- **Tail scenario description**: What drives the risk
- **Assumption logging**: All inputs tracked
- **Model uncertainty flags**: Known limitations highlighted

### Validation & Governance
- **Stress testing**: 4 extreme scenarios
- **Backtesting**: Forecast vs actual comparison
- **CVaR compliance tracking**: Historical performance
- **Audit trail**: Complete decision history

---

## 🎯 Use Cases

### 1. Real-Time Portfolio Balancing
- 15-minute interval decisions
- Risk-aware position management
- Automated mitigation selection

### 2. Risk Governance
- CVaR limit monitoring
- Escalation management
- Compliance reporting

### 3. Strategic Planning
- Stress testing portfolio resilience
- Flexibility asset sizing
- Hedge strategy optimization

### 4. Performance Analysis
- Backtesting accuracy
- Action effectiveness
- Cost-benefit analysis

---

## 📈 Dashboard Guide

### Forecast Team Interface

1. **Navigate to**: Forecast Data Input
2. **Enter probabilistic forecasts**:
   - Demand (P10, P50, P90)
   - PV generation (P10, P50, P90)
   - Wind generation (P10, P50, P90)
3. **Set market prices**:
   - Day-Ahead, Intraday, ReBAP±
4. **Validate forecast**
5. **Submit to Management**

### Management Interface

1. **Review validated forecast**
2. **Execute Decision Engine** (runs Monte Carlo)
3. **Review decision output**:
   - Risk state and metrics
   - Recommended action
   - CVaR/VaR values
   - Tail scenario analysis
4. **View loss distribution**
5. **Monitor governance compliance**

---

## 🔧 Configuration

### Risk Appetite Settings

```python
risk_appetite = RiskAppetite(
    cvar_limit_eur=1000.0,           # Max CVaR per interval
    var_limit_eur=800.0,             # Max VaR per interval
    confidence_level=0.95,           # 95% or 99%
    lambda_risk_aversion=2.0,        # Risk aversion parameter
    hold_threshold_mw=2.0,           # HOLD if |residual| < 2 MW
    watch_threshold_mw=5.0,          # WATCH if 2-5 MW
    action_threshold_mw=10.0,        # ACTION if > 10 MW
    manual_override_threshold_eur=5000.0,
    automated_action_max_eur=2000.0
)
```

### Monte Carlo Settings

```python
engine = DecisionEngine(
    n_scenarios=10000,    # Number of scenarios
    random_seed=42        # For reproducibility
)
```

### Correlation Structure

```python
correlations = CorrelationMatrix(
    demand_pv=-0.3,      # Demand-PV anticorrelation
    demand_wind=0.1,     # Demand-Wind weak correlation
    pv_wind=-0.2         # PV-Wind anticorrelation
)
```

---

## 📋 Files for Streamlit Deployment

### Required Files (5):
1. **sere_core.py** - Core engine
2. **sere_dashboard.py** - Main app (entry point)
3. **sere_database.py** - Database module
4. **sere_validation.py** - Validation module
5. **requirements.txt** - Dependencies

### Optional Files (3):
6. **README.md** - Documentation
7. **.gitignore** - Git exclusions
8. **.streamlit/config.toml** - Streamlit settings

### Deployment Steps

1. **Create GitHub repository**
2. **Upload all 5 required files**
3. **Go to**: https://streamlit.io/cloud
4. **Deploy with**:
   - **Main file**: `sere_dashboard.py`
   - **Python version**: 3.11+
   - **Repository**: your-username/sere-app

---

## 🧪 Testing

### Core Engine Test
```bash
python sere_core.py
```
**Expected**: Decision output with CVaR, VaR, risk state, and recommended action

### Dashboard Test
```bash
streamlit run sere_dashboard.py
```
**Expected**: Web interface at http://localhost:8501

### Validation Test
```bash
python sere_validation.py
```
**Expected**: Stress test results, backtesting metrics, governance report

### Database Test
```bash
python sere_database.py
```
**Expected**: Decision stored, history retrieved, compliance calculated

---

## 📊 Example Output

```
================================================================================
DECISION OUTPUT
================================================================================

Timestamp: 2026-01-29 08:00:00
Risk State: ACTION
Scenario Type: SHORTAGE
Confidence: 68.5%

--- RISK METRICS ---
Expected Loss: €713.28
VaR (95%): €2,630.22
CVaR (95%): €3,505.78
Risk Energy: €7,724.84
Residual: P10=-26.4, P50=0.1, P90=26.2 MW
Prob(Shortage): 50.1%, Prob(Surplus): 49.9%

--- RECOMMENDED ACTION ---
Action: INTRADAY_BUY
Volume: 15.0 MW
Cost: €562.50
CVaR Reduction: €2,150.00
Feasible: True
Rationale: Buy 15.0 MW intraday to reduce shortage risk

--- EXPLAINABILITY ---
Trigger: ACTION: CVaR €3,506 > limit €1,000
Rationale: Selected INTRADAY_BUY: 15.0 MW. Reduces CVaR by €2,150 at cost €563.
Tail Scenarios: Worst 5% scenarios: 500 cases with mean loss €3,506, max €7,727.
Dominated by demand=121.9 MW, PV=13.1 MW, wind=21.2 MW.

--- GOVERNANCE BREACH ---
Manual Override Required: NO
Escalation Required: YES
Details: CVaR (€3,506) exceeds limit (€1,000)
================================================================================
```

---

## 🔐 Governance Framework

### Risk States
- **HOLD**: CVaR within limits, no action needed
- **WATCH**: Approaching limits (75% of CVaR limit)
- **ACTION**: CVaR exceeds limit, mitigation required
- **CRITICAL**: CVaR >> limit, immediate escalation

### Escalation Triggers
- CVaR > limit
- VaR > limit
- Action cost > automated limit
- CRITICAL risk state

### Compliance Monitoring
- CVaR compliance rate (%)
- Escalation rate (%)
- Manual override rate (%)
- Action distribution

---

## 🎓 Technical Details

### Monte Carlo Implementation
- **Sampling**: Multivariate normal distribution
- **Correlation**: Cholesky decomposition
- **Scenarios**: Independent draws
- **Loss calculation**: Price × volume × 0.25h (15-min intervals)

### CVaR Calculation
```
VaR(α) = Percentile(losses, α)
CVaR(α) = E[loss | loss ≥ VaR(α)]
```
Where α = 0.95 (95% confidence)

### Risk Energy Formula
```
Risk Energy = E[loss] + λ · CVaR(95%)
```
Where λ = 2.0 (risk aversion parameter)

### Action Ranking
```
Score = CVaR_reduction / Cost
```
Select action with highest score among feasible options

---

## 📚 References

### Academic Foundation
- **CVaR**: Rockafellar & Uryasev (2000) - "Optimization of Conditional Value-at-Risk"
- **Monte Carlo**: Glasserman (2004) - "Monte Carlo Methods in Financial Engineering"
- **Risk Management**: Jorion (2006) - "Value at Risk: The New Benchmark for Managing Financial Risk"

### Energy Portfolio
- **Balancing**: Morales et al. (2014) - "Integrating Renewables in Electricity Markets"
- **Flexibility**: Zerrahn & Schill (2017) - "On the Representation of Demand-Side Management"

---

## 🤝 Support

### Documentation
- **User Guide**: See SERE_USER_GUIDE.md (to be created)
- **API Reference**: See docstrings in sere_core.py
- **Examples**: See test functions in each module

### Community
- **Issues**: Report bugs and request features
- **Discussions**: Share use cases and improvements
- **Contributions**: Pull requests welcome

---

## 📄 License

Production-ready software for energy risk management.

---

## 🏆 Key Differences from EPBDA

| Feature | EPBDA | SERE |
|---------|-------|------|
| **Simulation** | Optional MC | Mandatory 10,000+ scenarios |
| **Risk Metric** | CVaR (analytical) | CVaR (Monte Carlo) |
| **Correlations** | Not modeled | Explicit correlation matrix |
| **Tail Analysis** | Limited | Full tail statistics |
| **Validation** | Basic | Comprehensive backtesting |
| **Stress Testing** | No | 4 scenarios |
| **Governance** | Basic | Full compliance tracking |
| **Database** | Simple | SQLite with full audit trail |
| **Explainability** | Good | Excellent with tail scenarios |

---

## 🚀 Roadmap

### Version 2.1 (Planned)
- [ ] Real-time data connectors
- [ ] Advanced visualization (3D loss surfaces)
- [ ] Multi-day horizon optimization
- [ ] Machine learning forecast enhancement

### Version 2.2 (Planned)
- [ ] Portfolio optimization
- [ ] Hedge strategy backtesting
- [ ] REST API for integration
- [ ] Mobile dashboard

---

## ✅ Status

**Production Ready** ✅

- [x] Core engine tested
- [x] Dashboard operational
- [x] Database functional
- [x] Validation complete
- [x] Documentation comprehensive
- [x] Ready for Streamlit deployment

---

**Built with Python, NumPy, SciPy, Streamlit, and expertise in quantitative risk management.**

**SERE - Because tail risk matters.**

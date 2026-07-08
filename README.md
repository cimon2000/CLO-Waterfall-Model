# CLO-Waterfall-Model
Author: Rishi Bhardwaj
Background: 2 years hands-on experience in CLO/CDO administration at BNY Mellon (Corporate Trust) and Apex Fund Services — working directly with syndicated loan portfolios, OC/IC test monitoring, and institutional clients including BlackRock.

What This Model Does
This Python model simulates the cashflow waterfall of a Collateralized Loan Obligation (CLO) — a structured finance vehicle that pools leveraged loans and issues multiple rated tranches to investors.

The model covers:

Capital structure setup — Class A (AAA) through Class E (BB) notes + Equity tranche
Interest waterfall — sequential payment from senior to junior, with fees deducted first
Principal waterfall — sequential amortisation based on scheduled repayments and recoveries
OC (Overcollateralization) tests — Pool Balance / Notes Balance must exceed trigger; breach diverts cash to deleverage senior tranche
IC (Interest Coverage) tests — Collateral interest income / Note interest expense must exceed trigger; breach redirects cashflow
Default and recovery simulation — Constant Default Rate (CDR) applied annually with haircut on recovery proceeds
Three scenario analysis — Base Case, Stress, Severe Stress (GFC-style)
Equity IRR calculation — residual distributions to equity investors
Charts — waterfall breakdown, OC test ratios, cumulative losses, equity distributions
Capital Structure
Tranche	Rating	Size ($M)	Spread (bps)	OC Trigger	IC Trigger
Class A	AAA	248.0	+145	123.5%	120.0%
Class B	AA	32.0	+185	117.5%	115.0%
Class C	A	20.0	+240	112.5%	110.0%
Class D	BBB	16.0	+330	108.5%	106.0%
Class E	BB	12.0	+550	104.5%	103.0%
Equity	NR	12.0	Residual	—	—
Total Pool		400.0			
Initial Overcollateralization = 122.0%
SOFR assumption = 5.30% | Weighted Average Spread = 3.85%

Scenario Assumptions
Scenario	CDR (Peak)	Recovery Rate	Equity IRR
Base Case	2.5%	65%	~13%
Stress	6.0%	55%	~11%
Severe Stress	9.0%	40%	~8%
How to Run
# Install dependencies
pip install numpy pandas matplotlib

# Run the model
python clo_waterfall_model.py
Outputs:

Full cashflow table printed to console (period by period)
OC and IC test results with pass/fail indicators
Scenario comparison table
clo_waterfall_charts.png — four charts saved locally
Key CLO Concepts Explained
OC Test (Overcollateralization):
Pool balance divided by the outstanding balance of all notes at or senior to the tested tranche. If this falls below the trigger (e.g. 123.5% for Class A), available interest is redirected to pay down the most senior outstanding note instead of flowing to junior tranches.

IC Test (Interest Coverage):
Collateral interest income divided by total interest due on notes at or senior to the tested tranche. A breach signals that the pool is not generating enough income to service senior obligations — cash is again diverted to deleverage.

Waterfall priority:
Senior fees → Class A interest → [OC/IC test] → Class B interest → [OC/IC test] → ... → Class E interest → Sub fees → Equity residual

CDR (Constant Default Rate):
Annual percentage of the loan pool assumed to default. Recovery rate determines what fraction of defaulted principal is recovered. Net loss = defaults × (1 − recovery rate).

SOFR:
Secured Overnight Financing Rate — replaced LIBOR as the floating rate benchmark for leveraged loans and CLO notes.

Real-World Context
This model reflects structures I worked with directly at:

BNY Mellon Corporate Trust (June 2024 – June 2025) — CLO/CDO administration, Solvas platform, ATE mapping, OC/IC test monitoring
Apex Fund Services (September 2025 – Present) — Syndicated loan operations, trade processing, portfolio setup, fund administration
Files
File	Description
clo_waterfall_model.py	Main model — run this
clo_waterfall_charts.png	Output charts
README.md	This file

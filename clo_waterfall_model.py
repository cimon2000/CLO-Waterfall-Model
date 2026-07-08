"""
================================================================================
CLO WATERFALL MODEL
================================================================================
Author  : Rishi Bhardwaj
GitHub  : github.com/rishibhardwaj
LinkedIn: linkedin.com/in/rishibhardwaj

Description:
    A Collateralized Loan Obligation (CLO) waterfall model that simulates
    interest and principal distributions across tranches, including:
        - OC (Overcollateralization) and IC (Interest Coverage) test mechanics
        - Sequential interest and principal waterfalls
        - Default and recovery scenarios
        - Three scenario analysis: Base / Stress / Severe Stress
        - Equity IRR calculation
        - Full output tables and charts

Background:
    A CLO is a structured finance vehicle that:
      1. Buys a pool of leveraged loans (the 'collateral')
      2. Issues multiple rated tranches (Class A → Equity) to fund the purchase
      3. Distributes loan cashflows via a priority waterfall:
             Senior tranches paid first → Junior tranches → Equity gets residual
      4. OC/IC tests protect senior noteholders — if breached, cashflow is
         redirected to pay down senior notes instead of flowing to junior/equity

    Real-world experience: BNY Mellon Corporate Trust / Apex Fund Services
    tools referenced: Solvas, ATE Dashboard, Deal Manager, GCM
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)


# ============================================================================
# SECTION 1: CLO DEAL PARAMETERS
# ============================================================================

class CLOTranche:
    """Represents a single note tranche in the CLO capital structure."""
    def __init__(self, name, rating, balance, spread_bps, oc_trigger, ic_trigger):
        self.name        = name
        self.rating      = rating
        self.balance     = balance        # $M principal
        self.spread_bps  = spread_bps     # spread over SOFR in basis points
        self.oc_trigger  = oc_trigger     # OC test trigger (e.g. 123.5 = 123.5%)
        self.ic_trigger  = ic_trigger     # IC test trigger (e.g. 120.0 = 120%)
        self.is_equity   = (rating == "NR")


class CLODeal:
    """
    Defines all parameters of the CLO deal.

    Capital Structure (total $400M collateral pool):
    ┌──────────┬────────┬──────────┬──────────────┐
    │ Tranche  │ Rating │ Size $M  │ % of Pool    │
    ├──────────┼────────┼──────────┼──────────────┤
    │ Class A  │ AAA    │  248.0   │  62.0%       │
    │ Class B  │ AA     │   32.0   │   8.0%       │
    │ Class C  │ A      │   20.0   │   5.0%       │
    │ Class D  │ BBB    │   16.0   │   4.0%       │
    │ Class E  │ BB     │   12.0   │   3.0%       │
    │ Equity   │ NR     │   12.0   │   3.0%       │
    ├──────────┼────────┼──────────┼──────────────┤
    │ TOTAL    │        │  340.0   │  85.0%       │
    └──────────┴────────┴──────────┴──────────────┘
    Overcollateralization = $400M pool / $340M notes = 117.6%
    """
    def __init__(self):
        self.deal_name            = "Rishi CLO 2024-1, Ltd."
        self.pool_balance         = 400.0    # $M total collateral
        self.sofr                 = 0.0530   # 5.30% SOFR
        self.was                  = 0.0385   # Weighted Avg Spread on loans (385 bps)
        self.reinvestment_period  = 4        # years of reinvestment
        self.deal_tenor           = 10       # total periods (years) to model
        self.senior_mgmt_fee      = 0.0015   # 15 bps
        self.sub_mgmt_fee         = 0.0020   # 20 bps
        self.admin_fee            = 0.0005   # 5 bps trustee/admin

        # Tranches — senior (Class A) to junior (Equity)
        self.tranches = [
            CLOTranche("Class A", "AAA", 248.0, 145,  123.5, 120.0),
            CLOTranche("Class B", "AA",   32.0, 185,  117.5, 115.0),
            CLOTranche("Class C", "A",    20.0, 240,  112.5, 110.0),
            CLOTranche("Class D", "BBB",  16.0, 330,  108.5, 106.0),
            CLOTranche("Class E", "BB",   12.0, 550,  104.5, 103.0),
            CLOTranche("Equity",  "NR",   12.0,   0,    0.0,   0.0),
        ]

    def notes_balance(self):
        """Total outstanding note balance (excludes equity)."""
        return sum(t.balance for t in self.tranches if not t.is_equity)

    def overcollateralization_ratio(self, pool_bal=None):
        """OC ratio = Pool Balance / Total Notes Balance."""
        pool = pool_bal if pool_bal else self.pool_balance
        return (pool / self.notes_balance()) * 100


# ============================================================================
# SECTION 2: OC AND IC TEST CALCULATIONS
# ============================================================================

def run_oc_test(pool_balance, tranches_above, trigger):
    """
    OC Test: Pool Balance / Sum of notes at-and-above this tranche >= trigger.
    If OC ratio < trigger → TEST FAILS → cashflow diverted to pay down senior notes.

    Args:
        pool_balance    : Current collateral pool balance ($M)
        tranches_above  : List of CLOTranche objects at or senior to test tranche
        trigger         : OC trigger level (e.g. 123.5 for 123.5%)

    Returns:
        (ratio, passed): ratio as float, passed as bool
    """
    notes_bal = sum(t.balance for t in tranches_above)
    if notes_bal == 0:
        return 999.9, True
    ratio = (pool_balance / notes_bal) * 100
    return ratio, ratio >= trigger


def run_ic_test(interest_income, tranches_above, sofr, trigger):
    """
    IC Test: Collateral Interest Income / Interest Due on notes at-and-above >= trigger.
    If IC ratio < trigger → TEST FAILS → cashflow diverted to pay down senior notes.

    Args:
        interest_income : Annual interest from collateral pool ($M)
        tranches_above  : List of CLOTranche objects at or senior to test tranche
        sofr            : Current SOFR rate
        trigger         : IC trigger level (e.g. 120.0 for 120%)

    Returns:
        (ratio, passed): ratio as float, passed as bool
    """
    interest_due = sum(t.balance * (sofr + t.spread_bps / 10000) for t in tranches_above)
    if interest_due == 0:
        return 999.9, True
    ratio = (interest_income / interest_due) * 100
    return ratio, ratio >= trigger


# ============================================================================
# SECTION 3: WATERFALL ENGINE
# ============================================================================

def run_waterfall(deal, pool_balance, cdr, recovery_rate):
    """
    Runs the full CLO waterfall for a single period.

    Waterfall priority (simplified INDENTURE order):
      1. Senior fees (management + admin)
      2. Class A interest
      3. Class A OC/IC test → if fail, redirect to pay down Class A principal
      4. Class B interest  (if OC/IC tests pass)
      5. Class B OC/IC test
      6. Class C interest  → Class C OC/IC test
      7. Class D interest  → Class D OC/IC test
      8. Class E interest  → Class E OC/IC test
      9. Subordinated management fee
     10. Equity distribution (residual)

    Args:
        deal          : CLODeal object
        pool_balance  : Current collateral pool balance ($M)
        cdr           : Constant Default Rate for this period (annual %)
        recovery_rate : Recovery on defaulted loans (%)

    Returns:
        dict with all cashflow items, OC/IC test results, tranche payments
    """
    sofr     = deal.sofr
    tranches = deal.tranches

    # --- Collateral cashflows ---
    defaults          = pool_balance * cdr
    recoveries        = defaults * recovery_rate
    net_loss          = defaults - recoveries
    performing_pool   = pool_balance - defaults
    interest_income   = performing_pool * (sofr + deal.was) + recoveries
    scheduled_princ   = performing_pool * 0.08   # ~8% annual amortisation assumed

    # --- Fees (paid before any tranche interest) ---
    senior_fee = pool_balance * (deal.senior_mgmt_fee + deal.admin_fee)
    available  = interest_income - senior_fee

    results = {
        "pool_balance"       : pool_balance,
        "defaults"           : defaults,
        "recoveries"         : recoveries,
        "net_loss"           : net_loss,
        "interest_income"    : interest_income,
        "senior_fee"         : senior_fee,
        "available_interest" : available,
    }

    oc_results = {}
    ic_results = {}
    tranche_interest = {}
    tranche_principal = {}
    oc_diversion = 0.0

    non_equity = [t for t in tranches if not t.is_equity]

    for i, tranche in enumerate(non_equity):
        tranches_at_or_above = non_equity[: i + 1]
        tranche_rate          = sofr + tranche.spread_bps / 10000
        interest_due          = tranche.balance * tranche_rate

        # Pay tranche interest (if available cash)
        interest_paid = min(available, interest_due)
        available    -= interest_paid
        tranche_interest[tranche.name] = interest_paid

        # OC Test
        oc_ratio, oc_pass = run_oc_test(
            performing_pool, tranches_at_or_above, tranche.oc_trigger
        )
        oc_results[tranche.name] = {"ratio": oc_ratio, "pass": oc_pass, "trigger": tranche.oc_trigger}

        # IC Test
        ic_ratio, ic_pass = run_ic_test(
            interest_income, tranches_at_or_above, sofr, tranche.ic_trigger
        )
        ic_results[tranche.name] = {"ratio": ic_ratio, "pass": ic_pass, "trigger": tranche.ic_trigger}

        # If either test fails → divert available cash to pay down Class A principal
        if not oc_pass or not ic_pass:
            divert          = available
            oc_diversion   += divert
            available       = 0.0
            # Reduce Class A balance (senior-most tranche gets paydown)
            tranches[0].balance = max(0, tranches[0].balance - divert)

    # --- Subordinated fee ---
    sub_fee   = min(available, pool_balance * deal.sub_mgmt_fee)
    available -= sub_fee

    # --- Principal waterfall (sequential — Class A first) ---
    principal_available = scheduled_princ - net_loss + oc_diversion
    for tranche in non_equity:
        princ_paid              = min(tranche.balance, max(0, principal_available))
        tranche.balance        -= princ_paid
        principal_available    -= princ_paid
        tranche_principal[tranche.name] = princ_paid

    # --- Equity (residual interest + leftover principal) ---
    equity_distribution = available + max(0, principal_available)

    results.update({
        "oc_test"              : oc_results,
        "ic_test"              : ic_results,
        "oc_diversion"         : oc_diversion,
        "tranche_interest"     : tranche_interest,
        "tranche_principal"    : tranche_principal,
        "sub_fee"              : sub_fee,
        "equity_distribution"  : equity_distribution,
        "performing_pool"      : performing_pool,
    })

    return results


# ============================================================================
# SECTION 4: MULTI-PERIOD SIMULATION
# ============================================================================

def run_scenario(scenario_name, cdr_schedule, recovery_rate):
    """
    Runs the full CLO waterfall simulation over the deal tenor.

    Args:
        scenario_name   : Label for this scenario (e.g. "Base Case")
        cdr_schedule    : List of annual CDR values (one per year)
        recovery_rate   : Recovery rate assumption (constant)

    Returns:
        (summary_df, test_df): period-by-period cashflow and test results
    """
    deal         = CLODeal()
    pool_balance = deal.pool_balance
    cashflows    = []
    test_rows    = []

    # Track equity cashflows for IRR
    equity_cfs   = [-deal.tranches[-1].balance]   # initial equity investment

    for yr in range(1, deal.deal_tenor + 1):
        cdr    = cdr_schedule[yr - 1] if yr - 1 < len(cdr_schedule) else cdr_schedule[-1]
        result = run_waterfall(deal, pool_balance, cdr, recovery_rate)

        equity_cfs.append(result["equity_distribution"])

        # Build cashflow summary row
        row = {
            "Year"              : yr,
            "Pool Balance $M"   : round(pool_balance, 2),
            "Defaults $M"       : round(result["defaults"], 2),
            "Net Loss $M"       : round(result["net_loss"], 2),
            "Interest Income $M": round(result["interest_income"], 2),
            "OC Diversion $M"   : round(result["oc_diversion"], 2),
            "Equity Dist. $M"   : round(result["equity_distribution"], 2),
        }
        for t in deal.tranches[:-1]:
            row[f"{t.name} Int $M"]  = round(result["tranche_interest"].get(t.name, 0), 2)
            row[f"{t.name} Princ $M"]= round(result["tranche_principal"].get(t.name, 0), 2)

        cashflows.append(row)

        # Build OC/IC test row
        test_row = {"Year": yr}
        for tranche_name, data in result["oc_test"].items():
            test_row[f"OC {tranche_name}"] = f"{data['ratio']:.1f}% {'✓' if data['pass'] else '✗'}"
        for tranche_name, data in result["ic_test"].items():
            test_row[f"IC {tranche_name}"] = f"{data['ratio']:.1f}% {'✓' if data['pass'] else '✗'}"

        test_rows.append(test_row)

        # Update pool balance for next period
        pool_balance = result["performing_pool"] * 0.92   # amortization

    # Equity IRR
    try:
        irr = np.irr(equity_cfs) if hasattr(np, "irr") else _calc_irr(equity_cfs)
    except Exception:
        irr = _calc_irr(equity_cfs)

    summary_df = pd.DataFrame(cashflows)
    test_df    = pd.DataFrame(test_rows)

    print(f"\n{'='*72}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"  CDR: {[f'{c*100:.1f}%' for c in cdr_schedule[:5]]}...")
    print(f"  Recovery Rate: {recovery_rate*100:.0f}%")
    print(f"  Estimated Equity IRR: {irr*100:.1f}%")
    print(f"{'='*72}")

    return summary_df, test_df, equity_cfs, irr


def _calc_irr(cashflows):
    """Newton-Raphson IRR solver (fallback if np.irr unavailable)."""
    rate = 0.1
    for _ in range(1000):
        npv  = sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))
        dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-10:
            break
        rate -= npv / dnpv
        if rate <= -1:
            return -0.999
    return rate


# ============================================================================
# SECTION 5: VISUALISATION
# ============================================================================

def plot_results(base_cf, stress_cf, severe_cf, base_tests, deal):
    """Generates four charts summarising CLO performance across scenarios."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CLO Waterfall Model — Rishi Bhardwaj (2024)", fontsize=14, fontweight="bold", y=0.98)

    colors = {"Base Case": "#2563EB", "Stress": "#D97706", "Severe Stress": "#DC2626"}

    # ── Chart 1: Equity distributions across scenarios ──────────────────────
    ax1 = axes[0, 0]
    for label, df in [("Base Case", base_cf), ("Stress", stress_cf), ("Severe Stress", severe_cf)]:
        ax1.plot(df["Year"], df["Equity Dist. $M"], marker="o", label=label,
                 color=colors[label], linewidth=2, markersize=5)
    ax1.set_title("Equity distributions by scenario ($M)", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("$M")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))

    # ── Chart 2: Interest waterfall stacked bar (Base Case, Year 1) ─────────
    ax2 = axes[0, 1]
    tranche_names = ["Class A", "Class B", "Class C", "Class D", "Class E"]
    year1_int     = [base_cf[f"{t} Int $M"].iloc[0] for t in tranche_names]
    equity_dist   = [base_cf["Equity Dist. $M"].iloc[0]]
    bar_colors    = ["#1e40af", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#f59e0b"]
    all_vals      = year1_int + equity_dist
    all_labels    = tranche_names + ["Equity"]
    ax2.bar(all_labels, all_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_title("Interest waterfall — Base Case, Year 1 ($M)", fontweight="bold")
    ax2.set_ylabel("$M")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(all_vals):
        ax2.text(i, v + 0.05, f"${v:.2f}M", ha="center", va="bottom", fontsize=8)

    # ── Chart 3: OC Test ratios over time (Base Case, Class A & Class D) ────
    ax3 = axes[1, 0]
    oc_a = []
    oc_d = []
    for _, row in base_tests.iterrows():
        oc_a.append(float(row["OC Class A"].replace("✓", "").replace("✗", "").replace("%", "").strip()))
        oc_d.append(float(row["OC Class D"].replace("✓", "").replace("✗", "").replace("%", "").strip()))

    years = base_tests["Year"].tolist()
    ax3.plot(years, oc_a, marker="s", label="Class A OC Ratio", color="#2563EB", linewidth=2)
    ax3.plot(years, oc_d, marker="^", label="Class D OC Ratio", color="#7c3aed", linewidth=2)
    ax3.axhline(y=123.5, color="#2563EB", linestyle="--", alpha=0.5, linewidth=1, label="Class A Trigger (123.5%)")
    ax3.axhline(y=108.5, color="#7c3aed", linestyle="--", alpha=0.5, linewidth=1, label="Class D Trigger (108.5%)")
    ax3.set_title("OC Test ratios over time — Base Case", fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("OC Ratio (%)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # ── Chart 4: Cumulative defaults & losses by scenario ───────────────────
    ax4 = axes[1, 1]
    for label, df in [("Base Case", base_cf), ("Stress", stress_cf), ("Severe Stress", severe_cf)]:
        cum_loss = df["Net Loss $M"].cumsum()
        ax4.plot(df["Year"], cum_loss, marker="o", label=label,
                 color=colors[label], linewidth=2, markersize=5)
    ax4.set_title("Cumulative net losses by scenario ($M)", fontweight="bold")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Cumulative Loss $M")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))

    plt.tight_layout()
    plt.savefig("/home/claude/clo_waterfall_charts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Charts saved → clo_waterfall_charts.png")


# ============================================================================
# SECTION 6: MAIN — RUN ALL SCENARIOS
# ============================================================================

def main():
    print("\n" + "="*72)
    print("  CLO WATERFALL MODEL — Rishi Bhardwaj")
    print("  Rishi CLO 2024-1, Ltd. | Pool: $400M | Tranches: A/B/C/D/E + Equity")
    print("="*72)

    deal = CLODeal()
    print(f"\n  SOFR: {deal.sofr*100:.2f}%  |  WAS: {deal.was*100:.2f}%  |  "
          f"Initial OC: {deal.overcollateralization_ratio():.1f}%")

    print("\n  Capital Structure:")
    print(f"  {'Tranche':<10} {'Rating':<6} {'Balance $M':>12} {'Spread (bps)':>14} "
          f"{'OC Trigger':>12} {'IC Trigger':>12}")
    print("  " + "-"*68)
    for t in deal.tranches:
        print(f"  {t.name:<10} {t.rating:<6} {t.balance:>12.1f} "
              f"{'N/A (equity)':>14}" if t.is_equity else
              f"  {t.name:<10} {t.rating:<6} {t.balance:>12.1f} "
              f"{t.spread_bps:>14} {t.oc_trigger:>11.1f}% {t.ic_trigger:>11.1f}%")
    print(f"  {'TOTAL NOTES':<10} {'':<6} {deal.notes_balance():>12.1f}")
    print(f"  {'POOL':<10} {'':<6} {deal.pool_balance:>12.1f}")

    # ── Scenario Definitions ─────────────────────────────────────────────────
    # Base Case:    2.5% CDR, 65% recovery — normal credit environment
    # Stress:       5.0% CDR, 55% recovery — moderate downturn
    # Severe Stress:9.0% CDR, 40% recovery — GFC-style severe recession

    base_cdr   = [0.025] * 10
    stress_cdr = [0.025, 0.035, 0.050, 0.060, 0.055, 0.040, 0.030, 0.025, 0.025, 0.025]
    severe_cdr = [0.040, 0.060, 0.090, 0.090, 0.080, 0.070, 0.060, 0.050, 0.040, 0.035]

    base_df,   base_tests,   base_ecf,   base_irr   = run_scenario("Base Case",     base_cdr,   0.65)
    stress_df, stress_tests, stress_ecf, stress_irr = run_scenario("Stress",        stress_cdr, 0.55)
    severe_df, severe_tests, severe_ecf, severe_irr = run_scenario("Severe Stress", severe_cdr, 0.40)

    # ── Print Cashflow Tables ─────────────────────────────────────────────────
    print("\n\n  BASE CASE — Period Cashflows ($M)")
    print("  " + "-"*72)
    display_cols = ["Year", "Pool Balance $M", "Defaults $M", "Net Loss $M",
                    "Interest Income $M", "OC Diversion $M", "Equity Dist. $M"]
    print(base_df[display_cols].to_string(index=False))

    print("\n\n  BASE CASE — OC/IC Test Results")
    print("  (✓ = Pass | ✗ = Fail → cash diverted to senior repayment)")
    print("  " + "-"*72)
    oc_cols = ["Year"] + [c for c in base_tests.columns if c.startswith("OC")]
    ic_cols = ["Year"] + [c for c in base_tests.columns if c.startswith("IC")]
    print("\n  OC Tests:")
    print(base_tests[oc_cols].to_string(index=False))
    print("\n  IC Tests:")
    print(base_tests[ic_cols].to_string(index=False))

    # ── Scenario Comparison ───────────────────────────────────────────────────
    print("\n\n  SCENARIO COMPARISON — Equity IRR & Total Distributions")
    print("  " + "-"*60)
    print(f"  {'Scenario':<20} {'Equity IRR':>12} {'Total Equity $M':>16} {'Avg Annual Dist':>16}")
    print("  " + "-"*60)
    for label, df, irr in [
        ("Base Case",     base_df,   base_irr),
        ("Stress",        stress_df, stress_irr),
        ("Severe Stress", severe_df, severe_irr),
    ]:
        total = df["Equity Dist. $M"].sum()
        avg   = total / len(df)
        print(f"  {label:<20} {irr*100:>11.1f}% {total:>15.2f}M {avg:>15.2f}M")

    # ── Charts ────────────────────────────────────────────────────────────────
    plot_results(base_df, stress_df, severe_df, base_tests, deal)

    print("\n\n  KEY CONCEPTS DEMONSTRATED IN THIS MODEL:")
    print("  " + "-"*60)
    concepts = [
        ("OC Test",         "Pool balance / notes balance must exceed trigger; "
                            "breach diverts cash to deleverage senior tranche"),
        ("IC Test",         "Collateral interest income / note interest expense "
                            "must exceed trigger; breach redirects cashflow"),
        ("Waterfall",       "AAA paid first, equity paid last — subordination "
                            "protects senior noteholders from credit losses"),
        ("CDR",             "Constant Default Rate = annual % of pool that defaults; "
                            "recovery determines net loss to the structure"),
        ("Equity IRR",      "Residual cashflows to equity after all notes paid; "
                            "target 12–18% IRR is typical for CLO equity"),
        ("SOFR",            "Replaced LIBOR as floating rate benchmark; "
                            "all note coupons = SOFR + tranche spread"),
    ]
    for term, explanation in concepts:
        print(f"  {term:<16}: {explanation}")

    print("\n" + "="*72)
    print("  Model complete. Files generated:")
    print("  → clo_waterfall_model.py   (this file — upload to GitHub)")
    print("  → clo_waterfall_charts.png (charts — include in README)")
    print("="*72 + "\n")


if __name__ == "__main__":
    main()

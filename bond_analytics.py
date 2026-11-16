import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CorporateBond:
    """
    Fixed Income Pricing & Risk Analytics Engine for Corporate Bonds.
    Computes Dirty/Clean Price, Yield-to-Maturity (YTM), Macaulay Duration,
    Modified Duration, Convexity, DV01, and Credit Spread over Risk-Free Benchmark.
    """

    def __init__(
        self,
        issuer: str,
        rating: str,
        sector: str,
        face_value: float,
        coupon_rate: float,
        maturity_years: float,
        market_price: float,
        benchmark_yield: float,
        freq: int = 2,
    ):
        self.issuer = issuer
        self.rating = rating
        self.sector = sector
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.market_price = market_price
        self.benchmark_yield = benchmark_yield
        self.freq = freq

    def price_from_yield(self, ytm: float) -> float:
        """Compute theoretical bond price given a Yield-to-Maturity."""
        n_periods = int(round(self.maturity_years * self.freq))
        coupon = (self.coupon_rate * self.face_value) / self.freq
        r = ytm / self.freq
        periods = np.arange(1, n_periods + 1)
        cashflows = np.full(n_periods, coupon)
        cashflows[-1] += self.face_value
        discount_factors = (1 + r) ** (-periods)
        return float(np.sum(cashflows * discount_factors))

    def compute_ytm(self, tol: float = 1e-8, max_iter: int = 200) -> float:
        """Solve for Yield-to-Maturity using the Newton-Raphson numerical method."""
        ytm = self.coupon_rate  # Initial guess
        h = 1e-6
        for _ in range(max_iter):
            f_val = self.price_from_yield(ytm) - self.market_price
            if abs(f_val) < tol:
                return ytm
            f_prime = (self.price_from_yield(ytm + h) - self.price_from_yield(ytm - h)) / (2 * h)
            if abs(f_prime) < 1e-12:
                break
            ytm -= f_val / f_prime
        return ytm

    def risk_metrics(self) -> dict:
        """Compute YTM, Credit Spread (bps), Macaulay/Modified Duration, Convexity, and DV01."""
        ytm = self.compute_ytm()
        credit_spread_bps = (ytm - self.benchmark_yield) * 10_000

        n_periods = int(round(self.maturity_years * self.freq))
        coupon = (self.coupon_rate * self.face_value) / self.freq
        r = ytm / self.freq
        periods = np.arange(1, n_periods + 1)
        times_years = periods / self.freq

        cashflows = np.full(n_periods, coupon)
        cashflows[-1] += self.face_value
        pv_cashflows = cashflows * ((1 + r) ** (-periods))
        price = np.sum(pv_cashflows)

        macaulay_duration = np.sum(times_years * pv_cashflows) / price
        modified_duration = macaulay_duration / (1 + r)
        convexity = np.sum(pv_cashflows * periods * (periods + 1)) / (
            price * (self.freq**2) * ((1 + r) ** 2)
        )
        dv01 = modified_duration * price * 0.0001

        return {
            "Issuer": self.issuer,
            "Rating": self.rating,
            "Sector": self.sector,
            "Maturity (Y)": self.maturity_years,
            "Coupon (%)": round(self.coupon_rate * 100, 2),
            "Clean Price": round(self.market_price, 2),
            "YTM (%)": round(ytm * 100, 2),
            "Benchmark (%)": round(self.benchmark_yield * 100, 2),
            "Credit Spread (bps)": round(credit_spread_bps, 1),
            "Mod. Duration": round(modified_duration, 2),
            "Convexity": round(convexity, 2),
            "DV01 ($)": round(dv01, 4),
        }


def build_sample_credit_universe() -> list[CorporateBond]:
    """Create a realistic European Investment Grade & High Yield corporate bond sample."""
    return [
        CorporateBond("LVMH Moët Hennessy", "AA-", "Luxury", 100, 0.0325, 5.0, 99.40, 0.0245),
        CorporateBond("TotalEnergies SE", "A+", "Energy", 100, 0.0375, 7.0, 98.80, 0.0255),
        CorporateBond("Sanofi SA", "AA", "Healthcare", 100, 0.0300, 4.0, 99.10, 0.0240),
        CorporateBond("BNP Paribas (Snr)", "A+", "Financials", 100, 0.04125, 6.0, 98.20, 0.0250),
        CorporateBond("Renault SA", "BB+", "Automotive", 100, 0.0550, 5.0, 96.50, 0.0245),
        CorporateBond("Valeo SE", "BB+", "Auto Parts", 100, 0.05875, 4.5, 95.80, 0.0242),
        CorporateBond("Accor SA", "BBB-", "Hospitality", 100, 0.0450, 5.5, 97.60, 0.0248),
        CorporateBond("Air France-KLM", "BB", "Airlines", 100, 0.0725, 4.0, 97.10, 0.0240),
    ]


def generate_charts(df: pd.DataFrame, bonds: list[CorporateBond]) -> None:
    """Generate clean institutional-style credit spread and duration-convexity charts."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Chart 1: Credit Spread by Issuer (IG vs HY color-coded)
    df_sorted = df.sort_values("Credit Spread (bps)", ascending=True)
    colors = [
        "#1f4e78" if r.startswith(("AA", "A", "BBB")) else "#c0392b"
        for r in df_sorted["Rating"]
    ]
    axes[0].barh(
        df_sorted["Issuer"] + " (" + df_sorted["Rating"] + ")",
        df_sorted["Credit Spread (bps)"],
        color=colors,
        height=0.6,
    )
    axes[0].set_title("Corporate Credit Spreads over Risk-Free Benchmark (bps)", fontweight="bold")
    axes[0].set_xlabel("Credit Spread (bps)")

    # Chart 2: Price-Yield Convexity Profile (IG vs HY comparison)
    yield_grid = np.linspace(0.01, 0.12, 100)
    for bond in [bonds[0], bonds[4]]:  # LVMH (AA-) vs Renault (BB+)
        prices = [bond.price_from_yield(y) for y in yield_grid]
        axes[1].plot(
            yield_grid * 100,
            prices,
            linewidth=2.2,
            label=f"{bond.issuer} ({bond.rating}, {bond.maturity_years}Y)",
        )
    axes[1].set_title("Bond Price-Yield Convexity Profile (IG vs. High Yield)", fontweight="bold")
    axes[1].set_xlabel("Yield-to-Maturity (%)")
    axes[1].set_ylabel("Bond Price (% of Par)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("credit_analytics_dashboard.png", dpi=300)
    print("\nSaved chart: credit_analytics_dashboard.png")


if __name__ == "__main__":
    bonds = build_sample_credit_universe()
    results = [b.risk_metrics() for b in bonds]
    df_results = pd.DataFrame(results)

    print("=== EUROPEAN CORPORATE BOND & CREDIT SPREAD ANALYTICS ===")
    print(df_results.to_string(index=False))

    df_results.to_csv("credit_spread_report.csv", index=False)
    generate_charts(df_results, bonds)
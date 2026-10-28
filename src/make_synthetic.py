from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def to_churn(raw, target_rate):
    """Turn a risk score into Yes/No labels with a chosen overall churn rate."""
    raw = raw + RNG.normal(0, 0.8, len(raw))      # noise keeps it realistic
    lo, hi = -20.0, 20.0
    for _ in range(60):                           # find intercept for target rate
        mid = (lo + hi) / 2
        if sigmoid(raw + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    p = sigmoid(raw + (lo + hi) / 2)
    return np.where(RNG.random(len(raw)) < p, "Yes", "No")


def make_saas(n=5000):
    plan = RNG.choice(["Basic", "Pro", "Enterprise"], n, p=[0.5, 0.35, 0.15])
    price = pd.Series(plan).map({"Basic": 29, "Pro": 99, "Enterprise": 299}).to_numpy()
    seats = np.where(plan == "Basic", RNG.integers(1, 6, n),
                     np.where(plan == "Pro", RNG.integers(5, 40, n), RNG.integers(30, 300, n)))
    tenure = RNG.integers(1, 61, n)
    contract = RNG.choice(["Monthly", "Annual"], n, p=[0.6, 0.4])
    logins = np.clip(RNG.normal(40, 20, n), 0, None).round()
    adoption = (RNG.beta(2.5, 2.5, n) * 100).round(1)
    tickets = RNG.poisson(1.5, n)
    nps = RNG.integers(0, 11, n)
    pay_fail = RNG.poisson(0.3, n)
    last_login = np.clip(RNG.exponential(6, n), 0, 60).round()
    billing = RNG.choice(["Card", "Invoice", "PayPal"], n, p=[0.55, 0.3, 0.15])

    raw = (1.0 * (contract == "Monthly") - 0.03 * tenure - 0.025 * adoption
           + 0.25 * tickets - 0.12 * nps + 0.6 * pay_fail + 0.05 * last_login
           - 0.01 * logins - 0.3 * (plan == "Enterprise"))

    return pd.DataFrame({
        "customer_id": [f"SAAS-{i:05d}" for i in range(n)],
        "plan": plan, "seats": seats, "MonthlyCharges": price,
        "tenure_months": tenure, "Contract": contract,
        "logins_last_30d": logins, "feature_adoption_pct": adoption,
        "support_tickets_90d": tickets, "nps_score": nps,
        "payment_failures_6m": pay_fail, "days_since_last_login": last_login,
        "billing_method": billing, "Churn": to_churn(raw, 0.22),
    })


def make_ott(n=5000):
    plan = RNG.choice(["Mobile", "Standard", "Premium"], n, p=[0.35, 0.45, 0.20])
    price = pd.Series(plan).map({"Mobile": 6, "Standard": 12, "Premium": 18}).to_numpy()
    tenure = RNG.integers(1, 49, n)
    cycle = RNG.choice(["Monthly", "Annual"], n, p=[0.75, 0.25])
    watch = np.clip(RNG.normal(25, 14, n), 0, None).round(1)
    change = np.clip(RNG.normal(0, 30, n), -100, 100).round(1)
    session = np.clip(RNG.normal(45, 20, n), 5, None).round()
    devices = RNG.integers(1, 6, n)
    inactive = np.clip(RNG.exponential(5, n), 0, 60).round()
    tickets = RNG.poisson(0.8, n)
    pay_fail = RNG.poisson(0.25, n)
    method = RNG.choice(["Card", "UPI", "Wallet", "Carrier billing"], n, p=[0.4, 0.3, 0.2, 0.1])

    raw = (0.7 * (cycle == "Monthly") - 0.03 * tenure - 0.05 * watch - 0.02 * change
           + 0.05 * inactive + 0.5 * pay_fail + 0.2 * tickets - 0.2 * devices)

    return pd.DataFrame({
        "customer_id": [f"OTT-{i:05d}" for i in range(n)],
        "plan": plan, "MonthlyCharges": price, "tenure_months": tenure,
        "BillingCycle": cycle, "watch_hours_30d": watch,
        "watch_hours_change_pct": change, "avg_session_min": session,
        "devices_count": devices, "days_since_last_active": inactive,
        "support_tickets_90d": tickets, "payment_failures_6m": pay_fail,
        "payment_method": method, "Churn": to_churn(raw, 0.25),
    })


if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    for name, fn in [("saas", make_saas), ("ott", make_ott)]:
        df = fn()
        df.to_csv(f"data/raw/{name}.csv", index=False)
        print(name, df.shape, "churn rate:", round((df["Churn"] == "Yes").mean(), 3))
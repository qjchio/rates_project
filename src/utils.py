import copy
import datetime
from datetime import date
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm


class YieldCurve:
    def __init__(
        self,
        start: datetime = date.today() - datetime.timedelta(days=10),
        end: datetime = date.today(),
    ):
        """
        Fetch data from FRED, build the US Treasury Yield Curve

        """
        self.start = start
        self.end = end

        series_ids = {
            "1M": "DGS1MO",
            "3M": "DGS3MO",
            "6M": "DGS6MO",
            "1Y": "DGS1",
            "2Y": "DGS2",
            "3Y": "DGS3",
            "5Y": "DGS5",
            "7Y": "DGS7",
            "10Y": "DGS10",
            "20Y": "DGS20",
            "30Y": "DGS30",
        }

        # -----------------------------
        # Fetch from FRED
        # -----------------------------
        df = pd.DataFrame()

        for label, fred_id in series_ids.items():
            df[label] = web.DataReader(fred_id, "fred", start, end)

        latest = df.iloc[-1].dropna()
        # self.df = df
        self.latest = latest

    def get_tenors(self):
        latest = self.latest
        tenors = [
            int(ten[:-1]) if ten.endswith("Y") else int(ten[:-1]) / 12
            for ten in list(latest.index)
        ]
        return tenors

    def get_yields(self):
        latest = self.latest
        yields = list(latest)
        return yields

    def get_tenors_yields(self):
        return {"tenors": self.get_tenors(), "yields": self.get_yields()}

    def parallel_shift(self, shift_in_bps: float):
        new_curve = copy.deepcopy(self)
        new_curve.latest += shift_in_bps / 100

        return new_curve

    def bump_curve(self, bumps: dict):
        new_curve = copy.deepcopy(self)

        latest_dct = new_curve.latest.to_dict()
        name = new_curve.latest.name

        for tenor, bp in bumps.items():
            if tenor not in latest_dct:
                raise KeyError(
                    f"Tenor {tenor} not present in yield curve, please use the correct tenors."
                )
            else:
                latest_dct[tenor] += bp / 100

        new_curve.latest = pd.Series(latest_dct, name=name)
        return new_curve

    def __str__(self):
        latest = self.latest
        date = latest.name
        # print("Latest yields:")
        # print(latest)
        # -----------------------------
        # Plot yield curve
        # -----------------------------
        plt.figure(figsize=(8, 5))
        plt.plot(latest.index, latest.values, marker="o")
        plt.title("US Treasury Yield Curve")
        plt.ylabel("Yield (%)")
        plt.xlabel("Maturity")
        plt.grid(True)
        plt.show()

        return f"US Treasury Yields as of {latest.name.date()}: \n{latest}"


class Bond:
    def __init__(self, maturity: float, coupon: float, notional: float = 1e6):
        self.period = 2  # semi annual payments
        self.face_value = 100  # face value

        self.maturity = maturity
        self.coupon = coupon
        self.notional = notional

    def get_price(self, yCurve: YieldCurve) -> float:

        dct = yCurve.get_tenors_yields()
        tenors = np.array(dct["tenors"])
        yields = np.array(dct["yields"])

        bond_disc = np.arange(start=self.maturity, stop=0, step=-0.5)[::-1]
        cashflows = [self.coupon * self.face_value] * (len(bond_disc) - 1) + [
            (1 + self.coupon) * self.face_value,
        ]
        discount_rates = (
            np.interp(
                bond_disc,
                tenors,
                yields,
            )
            / 100
        )
        res = 0
        i = 1
        for cf, d in zip(cashflows, discount_rates):
            res += cf / ((1 + d / self.period) ** i)
            i += 1

        # print(bond_disc, cashflows, discount_rates)
        return res

    def get_total_value(self, yCurve: YieldCurve):
        return self.get_price(yCurve) * self.notional / self.face_value

    def __str__(self):
        return (
            f"US Bond with maturity: {self.maturity} years, coupon rate: {self.coupon}"
        )


class Portfolio:
    def __init__(self):
        self.bonds = []

    def add_bonds(self, bond: Bond):
        self.bonds.append(bond)

    def get_total_value(self, yCurve: YieldCurve):
        res = 0
        for b in self.bonds:
            res += b.get_total_value(yCurve)
        return res

    def get_value_by_tenor(self, yCurve: YieldCurve, bucket: bool = False):
        tenors = [b.maturity for b in self.bonds]

        if bucket:
            tenors = [round(x, 0) for x in tenors]

        val = [b.get_total_value(yCurve) for b in self.bonds]

        dct = {}

        for k, v in zip(tenors, val):
            if k in dct:
                dct[k] += v
            else:
                dct[k] = v

        # print({k:v for k,v in zip(tenors, val)})

        return pd.Series(dct.values(), index=dct.keys())


def compare_curves(curves: Dict[str, YieldCurve]) -> None:

    plt.figure(figsize=(8, 5))

    for k, cur in curves.items():
        plt.plot(cur.latest.index, cur.latest.values, label=k)
    # plt.plot(new_curve.latest.index, new_curve.latest.values, marker="o")

    plt.title("US Treasury Yield Curve")
    plt.ylabel("Yield (%)")
    plt.xlabel("Maturity")
    plt.grid(True)
    plt.legend()
    plt.show()
    pass


def get_summary(
    portfolio: Portfolio,
    curves: Dict[str, YieldCurve],
) -> pd.DataFrame:

    base_curve = curves["YieldCurve"]
    original_pnl = portfolio.get_total_value(base_curve)
    date = base_curve.latest.name
    dct = {}

    for k, curve in curves.items():
        if k != "YieldCurve":
            name = f"{k} % Change in PnL"
            name = "_".join(name.split("_")[1:])
            dct[name] = [(portfolio.get_total_value(curve) / original_pnl - 1) * 100]

    return pd.DataFrame(dct, index=[date]).round(3)


def get_heatmap(
    portfolio: Portfolio, curves: Dict[str, YieldCurve], bucket: bool = False
) -> None:

    base_curve = curves["YieldCurve"]
    original_pnl = portfolio.get_value_by_tenor(base_curve, bucket)
    date = base_curve.latest.name

    tenors = list(original_pnl.index)

    N = len(curves) - 1
    TENORS = tenors * N
    pnl = []
    SCENARIO = []

    for k, curve in curves.items():
        if k != "YieldCurve":
            name = "_".join(k.split("_")[1:])
            SCENARIO.extend([name] * len(tenors))
            pnl.append(
                np.array(
                    (portfolio.get_value_by_tenor(curve, bucket) / original_pnl - 1)
                    * 100
                )
            )

    # print(TENORS)
    # print(SCENARIO)
    # print(pnl)
    df = pd.DataFrame(
        {
            "Maturity": TENORS,
            "Scenario": SCENARIO,
            "% PnL change": np.concatenate(pnl),
        }
    )

    # Pivot to get Scenario vs Maturity grid
    pivot = df.pivot(index="Scenario", columns="Maturity", values="% PnL change")
    pivot = pivot.round(2)

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, cmap="RdYlGn", center=0, cbar_kws={"label": "PnL %"})
    plt.title("Portfolio % P&L Under Yield Curve Scenarios")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Scenario")
    plt.show()

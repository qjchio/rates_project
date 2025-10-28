from utils import *

lst_of_bonds = [
    Bond(1.7, 0.036),
    Bond(2, 0.04),
    Bond(3, 0.043),
    Bond(5, 0.045),
    Bond(5.5, 0.0453),
    Bond(7, 0.0476),
    Bond(10, 0.053),
]

yc = YieldCurve()
portfolio = Portfolio()
[portfolio.add_bonds(b) for b in lst_of_bonds]
print("Yield Curve generated")


yc_parallel_add_10 = yc.parallel_shift(10)
yc_parallel_minus_10 = yc.parallel_shift(-10)
yc_steepener = yc.bump_curve({"2Y": -10, "3Y": -5, "5Y": 2, "7Y": 4, "10Y": 6})
yc_flattener = yc.bump_curve({"2Y": 10, "3Y": 5, "5Y": -1, "7Y": -4, "10Y": -6})
curves = {
    "YieldCurve": yc,
    "YieldCurve_parallel_+10bps": yc_parallel_add_10,
    "YieldCurve_parallel_-10bps": yc_parallel_minus_10,
    "YieldCurve_steepener": yc_steepener,
    "YieldCurve_flattener": yc_flattener,
}


# get some plots

# compare_curves(curves)
# get_summary(portfolio, curves)
# get_heatmap(portfolio, curves)
# get_heatmap(portfolio, curves, True)

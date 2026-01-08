
import statsmodels.api as sm

def clv_regression(df):
    X = sm.add_constant(df["clv"])
    y = df["roi"]
    return sm.OLS(y, X).fit().summary()

"""
motor_gbm.py — Parte matemática (sin dependencias de interfaz)
GBM multiactivo correlacionado (Normal / t-Student), VaR-CVaR, Black-Scholes vs MC, martingala.

Uso independiente:
    from motor_gbm import simular
    R = simular(("SPY", "QQQ", "AAPL", "MSFT", "GLD"), 10000, 5)
"""
import numpy as np
import scipy.stats as stats
import yfinance as yf

STEPS = 252          # 1 año en días hábiles
T = 1.0
BONOS13W = "^IRX"


def gbm(So, sigma, L, n_sims, steps, T, r, nu=None):
    dt = T / steps
    n_activos = len(So)

    rng = np.random.default_rng(42)
    if nu is None:
        Z = rng.standard_normal(size=(n_sims, steps, n_activos))
    else:
        Z = rng.standard_t(df=nu, size=(n_sims, steps, n_activos)) / np.sqrt(nu / (nu - 2))

    W = Z @ L.T

    # Ecuación exacta del GBM en log-retornos
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    acumulados = np.cumsum(log_returns, axis=1)

    # Añadir instante inicial t = 0
    zeros = np.zeros((n_sims, 1, n_activos))
    F = np.concatenate([zeros, acumulados], axis=1)
    return So * np.exp(F)


def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def descargar(tickers: tuple):
    Close = yf.download(list(tickers), period="1y", progress=False)["Close"]
    # yfinance devuelve las columnas en orden alfabético: se reordenan
    # para que cada resultado corresponda al ticker correcto.
    Close = Close[list(tickers)].dropna()
    r = yf.Ticker(BONOS13W).history(period="5d")["Close"].iloc[-1] / 100
    return Close, float(r)


def simular(tickers: tuple, N_sims: int, nu: int):
    Close, r = descargar(tickers)
    steps = STEPS

    So = Close.iloc[-1].to_numpy()
    retornosLog = np.diff(np.log(Close.to_numpy()), axis=0)
    sigma = retornosLog.std(axis=0, ddof=1) * np.sqrt(steps)

    corr = np.corrcoef(retornosLog, rowvar=False)
    L = np.linalg.cholesky(corr)

    precios = gbm(So, sigma, L, N_sims, steps, T, r)
    precios_t = gbm(So, sigma, L, N_sims, steps, T, r, nu=nu)

    martingala = precios[:, -1, :].mean(axis=0) * np.exp(-r * T)
    errores = np.abs(martingala - So) / So * 100

    media = np.mean(precios, axis=0)
    Mediana = np.percentile(precios, 50, axis=0)
    p5, p25, p75, p95 = np.percentile(precios, [5, 25, 75, 95], axis=0)

    S_T_norm = precios[:, -1, :]
    S_T_stud = precios_t[:, -1, :]
    perdidas_norm = So - S_T_norm
    perdidas_stud = So - S_T_stud

    var95 = np.percentile(perdidas_norm, 95, axis=0)
    Cvar = np.array([perdidas_norm[perdidas_norm[:, i] >= var95[i], i].mean() for i in range(len(So))])
    var95_t = np.percentile(perdidas_stud, 95, axis=0)
    Cvar_t = np.array([perdidas_stud[perdidas_stud[:, i] >= var95_t[i], i].mean() for i in range(len(So))])

    K = So
    bs_call = black_scholes_call(So, K, r, sigma, T)
    mc_call_norm = (np.exp(-r * T) * np.maximum(S_T_norm - K, 0)).mean(axis=0)
    mc_call_stud = (np.exp(-r * T) * np.maximum(S_T_stud - K, 0)).mean(axis=0)

    return dict(Close=Close, r=r, So=So, retornosLog=retornosLog, sigma=sigma, corr=corr, L=L,
                precios=precios, precios_t=precios_t, martingala=martingala, errores=errores,
                media=media, Mediana=Mediana, p5=p5, p25=p25, p75=p75, p95=p95,
                perdidas_norm=perdidas_norm, perdidas_stud=perdidas_stud,
                var95=var95, Cvar=Cvar, var95_t=var95_t, Cvar_t=Cvar_t,
                bs_call=bs_call, mc_call_norm=mc_call_norm, mc_call_stud=mc_call_stud)


if __name__ == "__main__":
    tick = ("SPY", "QQQ", "AAPL", "MSFT", "GLD")
    R = simular(tick, 10000, 5)
    np.set_printoptions(precision=4, suppress=True)
    print("Tickers     :", tick)
    print("r           :", R["r"])
    print("sigma       :", R["sigma"])
    print("Err. mart. %:", R["errores"])
    print("VaR95 N     :", R["var95"], "| CVaR N:", R["Cvar"])
    print("VaR95 t     :", R["var95_t"], "| CVaR t:", R["Cvar_t"])
    print("Call BS     :", R["bs_call"])
    print("Call MC N   :", R["mc_call_norm"], "| MC t:", R["mc_call_stud"])

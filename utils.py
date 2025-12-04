import math
import re
import csv
from typing import Optional
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
from scipy.optimize import curve_fit, OptimizeWarning


filepath = rf"{os.getcwd()}\data"


def processRawData(dataraw: str) -> None:
    r1 = re.compile(r"(\d)( @){1}([A-Z])")
    r2 = re.compile(r"(\d)(  ){1}([A-Z])")
    r3 = re.compile(" {3,}")
    r4 = re.compile("([0-9]) (P)[A-Z0-9]*,")
    r5 = re.compile("([0-9])[ A-Z]+.*,")
    r6 = re.compile("0 (Sch)|(SchP)")
    r7 = re.compile("  ")
    r8 = re.compile("$")
    r9 = re.compile("(,){4,}")
    s1 = r"\g<1>,1,\g<3>"
    s2 = r"\g<1>,0,\g<3>"
    s3 = ","
    s4 = r"\g<1>,\g<2>,"
    s5 = r"\g<1>,"
    s6 = "0"
    s7 = " "
    s8 = ",,,"
    s9 = ",,,,"

    edited = []

    with open(dataraw, "r") as f:
        reader = csv.reader(f, delimiter="]")
        for row in reader:
            patterns = [r1, r2, r3, r4, r5, r6, r7, r8, r9]
            substitutions = [s1, s2, s3, s4, s5, s6, s7, s8, s9]

            x = row[0]
            for pattern, substitution in zip(patterns, substitutions):
                x = pattern.sub(substitution, x)
            edited.append(x)

    with open(dataraw, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter="]", quotechar=">", quoting=csv.QUOTE_MINIMAL
        )
        for i in range(len(edited)):
            csvwriter.writerow([edited[i]])


def log(logfile: str, *text) -> None:
    """
    write log event to log file
    """
    for t in text:
        if isinstance(t, list):
            t = ",".join(str(i) for i in t)
        else:
            t = str(t)

    text = ",".join(str(t) for t in text)

    with open(logfile, "a") as f:
        f.write(text + "\n")


def avg(x):
    s = 0
    n = 0

    for i in x:
        if i is None:
            s = s
            n = n
        else:
            s += i
            n += 1

    if n == 0:
        return 0
    else:
        return s / n


def brier(accuracies: list[tuple[int, int, float]], **kwargs) -> dict[str, float]:
    if not accuracies:
        return {"rel": 0.0, "res": 0.0, "unc": 0.0}

    platt: Optional[tuple[float, float]] = kwargs.get("platt", None)
    A, B = platt or (0.0, 0.0)

    outcomes = [
        (
            i
            if not platt
            else [
                1 if platt_scale(i[2], A, B) > 0.5 else 0,
                float(platt_scale(i[2], A, B)),
            ]
        )
        for i in accuracies
    ]

    base_rate = 0.5
    N = len(outcomes)
    rel = 0.0
    res = 0.0
    unc = base_rate * (1 - base_rate)

    for k in range(0, 101):
        p_k = k / 100
        o = [a[1] for a in outcomes if round(a[2], 2) == p_k]
        if not o:
            continue
        o_avg = avg(o)
        rel += len(o) * (p_k - o_avg) ** 2
        res += len(o) * (o_avg - base_rate) ** 2

    brier = {"rel": rel / N, "res": res / N, "unc": unc}
    return brier


def platt_scale(f: float, a: float, b: float) -> float:
    return 1 / (1 + np.exp(-(a * f + b)))


def platt_scaling(accuracies: list[tuple[int, int, float]]) -> tuple[float, float]:
    x = np.array([np.clip(i[2], 0.01, 0.99) for i in accuracies]).reshape(-1, 1)
    y = np.array([i[1] for i in accuracies]).reshape(-1, 1).ravel()
    platt_scaler = LogisticRegression(penalty="l2", C=1000.0, solver="liblinear")
    platt_scaler.fit(x, y)
    A = platt_scaler.coef_[0][0]
    B = platt_scaler.intercept_[0]
    return float(A), float(B)


def gaussian(x, a, z, w):
    return a * np.exp(-(((x - z) / w) ** 2))


def getcoeff(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
) -> tuple[float, float, float]:
    """Solves for quadratic coefficients"""

    if y1 > 1:
        raise ValueError(f"y1 > 1")

    if y3 > 1:
        raise ValueError(f"y3 > 1")
    x = [x1, x2, x3]
    x = np.asarray(x).ravel()
    y = [y1, y2, y3]
    y = np.asarray(y).ravel()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        f = curve_fit(gaussian, x, y, [1, x2, (x3 - x1) / 2], gtol=0.1)[0]
        return f[0], f[1], f[2]


def getroots(a, b, c, d, e, f):
    """Get the roots of the point spread"""

    m = (e * c**2 + b * f**2) / (c**2 + f**2)
    m = a * d * np.exp(-(((m - b) / c) ** 2) - ((m - e) / f) ** 2)
    mark = m / 200

    if m == 0:
        m = a * d

    try:
        r1 = np.floor(
            (
                2 * b * f * f
                + 2 * c * c * e
                - c
                * f
                * np.sqrt(
                    8 * b * e
                    - 4 * e * e
                    - 4 * c * c * math.log(m / a / d)
                    - 4 * f * f * math.log(m / a / d)
                    - 4 * b * b
                    + 4 * c * c * math.log(200)
                    + 4 * f * f * math.log(200)
                )
            )
            / (2 * (c * c + f * f))
        )
    except ValueError:
        r1 = np.floor((2 * b * f * f + 2 * c * c * e - c * f) / (2 * (c * c + f * f)))

    try:
        r2 = np.ceil(
            (
                2 * b * f * f
                + 2 * c * c * e
                + c
                * f
                * np.sqrt(
                    8 * b * e
                    - 4 * e * e
                    - 4 * c * c * math.log(m / a / d)
                    - 4 * f * f * math.log(m / a / d)
                    - 4 * b * b
                    + 4 * c * c * math.log(200)
                    + 4 * f * f * math.log(200)
                )
            )
            / (2 * (c * c + f * f))
        )
    except ValueError:
        r2 = np.ceil((2 * b * f * f + 2 * c * c * e + c * f) / (2 * (c * c + f * f)))

    return max(0, r1), max(3, r2)

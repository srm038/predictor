import re
import csv
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
import os


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

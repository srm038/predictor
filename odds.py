import os
from dotenv import load_dotenv
from math import prod, sqrt
from game import Game
from sport import Sport
from predictor import SportCode, loadSport
import numpy as np


def runParlays(analyses: list[dict]) -> list[dict]:

    parlays = []

    for i in range(len(analyses)):
        for j in range(i + 1, len(analyses)):
            parlays.append({"games": (analyses[i], analyses[j])})

    for i in range(len(analyses)):
        for j in range(i + 1, len(analyses)):
            for k in range(j + 1, len(analyses)):
                parlays.append({"games": (analyses[i], analyses[j], analyses[k])})

    for i in range(len(analyses)):
        for j in range(i + 1, len(analyses)):
            for k in range(j + 1, len(analyses)):
                for l in range(k + 1, len(analyses)):
                    parlays.append(
                        {"games": (analyses[i], analyses[j], analyses[k], analyses[l])}
                    )

    for p in parlays:
        tp = []
        tip = []
        tev = []
        tor = []
        pick = []
        skip = False
        for g in p["games"]:
            if g["EV"][0] > 0:
                pick.append(0)
            elif g["EV"][1] > 0:
                pick.append(1)
            else:
                skip = True
                continue
            tp.append(g["game"].w1 if pick[-1] == 0 else g["game"].w2)
            tip.append(g["ip"][pick[-1]])
            tev.append(g["EV"][pick[-1]])
            tor.append(g["OR"][pick[-1]])
        if not skip:
            p.update({"tp": prod(tp)})
            p.update({"tip": prod(tip)})
            p.update({"tev": sum(tev)})
            p.update({"tor": prod(tor)})
            p.update({"pick": pick})
            p.update({"f*": (p["tev"] / (p["tor"] - 1)) * kelly_fraction})

    parlays = [p for p in parlays if "tp" in p]

    for p in sorted(parlays, key=lambda x: x["tev"] * sqrt(x["tp"]), reverse=True)[:10]:
        if p["tev"] <= 1.0:
            continue
        picks = [
            g["game"].t1 if pick == 0 else g["game"].t2
            for g, pick in zip(p["games"], p["pick"])
        ]
        if p["f*"] * bankroll >= 0.1:
            print(
                f"Bet ${betAmount(p['f*'], bankroll):0.2f} ({p['tev']:0.2f}, {p['tp']:0.0%}) {' + '.join(picks)}"
            )

    return parlays


class BettingEngine:
    def __init__(self, sportCode: SportCode, year: int) -> None:
        self.sport = loadSport(sportCode, year)
        self.loadConfig()
        self.loadOdds()
        self.analyze()

    def loadConfig(self):

        load_dotenv(override=True)

        self.kelly_fraction = float(os.getenv("f") or 0.25)
        self.bankroll = float(os.getenv("BANKROLL") or 0)
        self.stabilizer = float(os.getenv("STABILIZER") or 1.0)

    def loadOdds(self):

        with open("odds.tsv", "r") as file:
            odds = [
                tuple(line.strip().split("\t"))
                for line in file.readlines()
                if line.strip() and not line.startswith("#")
            ]
            odds = [
                Odds(
                    t1,
                    t2,
                    int(o1),
                    int(o2),
                    sport=self.sport,
                    kelly_fraction=self.kelly_fraction,
                    bankroll=self.bankroll,
                    stabilizer=self.stabilizer,
                )
                for t1, t2, o1, o2 in odds
            ]

        self.odds = odds

    def analyze(self) -> None:
        self.analyses = [odds.analyze() for odds in self.odds]
        for a in sorted(self.odds, key=lambda x: max(x.f_star), reverse=True):
            print(a)


class Odds:
    def __init__(
        self,
        t1: str,
        t2: str,
        o1: int,
        o2: int,
        sport: Sport,
        kelly_fraction: float,
        bankroll: float,
        stabilizer: float,
    ):
        self.t1 = t1
        self.t2 = t2
        self.o1 = o1
        self.o2 = o2
        self.sport = sport
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll
        self.stabilizer = stabilizer

    def calcIP(self) -> tuple[float, float]:
        return (
            -self.o1 / (100 - self.o1) if self.o1 < 0 else 100 / (self.o1 + 100),
            -self.o2 / (100 - self.o2) if self.o2 < 0 else 100 / (self.o2 + 100),
        )

    def calcOR(self) -> tuple[float, float]:
        return (
            1 + (-100 / self.o1 if self.o1 < 0 else self.o1 / 100),
            1 + (-100 / self.o2 if self.o2 < 0 else self.o2 / 100),
        )

    def calcEV(self) -> tuple[float, float]:
        w1 = self.game.w1
        w2 = self.game.w2
        return (w1 * (self.OR[0] - 1) - w2, w2 * (self.OR[1] - 1) - w1)

    def calcFStar(self) -> tuple[float, float]:
        return (
            self.EV[0] / (self.OR[0] - 1) * self.kelly_fraction,
            self.EV[1] / (self.OR[1] - 1) * self.kelly_fraction,
        )

    def analyze(self):
        self.game = Game(
            t1=self.t1,
            t2=self.t2,
            h1=False,
            h2=True,
            p1=None,
            p2=None,
            sport=self.sport,
        )
        self.game.w()
        self.ip = self.calcIP()
        self.OR = self.calcOR()
        self.EV = self.calcEV()
        self.f_star = self.calcFStar()

        self.warning = (self.EV[0] > 0 and self.o1 > 0) or (
            self.EV[1] > 0 and self.o2 > 0
        )

        return

    def betAmount(self) -> tuple[float, float]:
        return np.clip(self.f_star, 0, 0.05, dtype=float) * self.bankroll

    def __str__(self) -> str:
        if self.EV[0] > 0.2:
            if self.f_star[0] * self.bankroll >= 0.1:
                return f"Bet ${self.betAmount()[0]:0.2f} ({self.EV[0]:0.2f}, {self.game.w1:0.0%}) on {self.game.t1} (@ {self.game.t2}) {'\uEA6C' if self.warning else ''}"
            elif self.EV[0] > 0.5 and self.game.w1 >= self.stabilizer:
                return f"Bet ${0.01 * self.bankroll:0.2f} ({self.EV[0]:0.2f}, {self.game.w1:0.0%}) on {self.game.t1} (@ {self.game.t2}) {'\uEA6C' if self.warning else ''} \uEAA5"
            if self.EV[1] > 0.2:
                if self.f_star[1] * self.bankroll >= 0.1:
                    return f"Bet ${self.betAmount()[1]:0.2f} ({self.EV[1]:0.2f}, {self.game.w2:0.0%}) on {self.game.t2} (v {self.game.t1}) {'\uEA6C' if self.warning else ''}"
            elif self.EV[1] > 0.5 and self.game.w2 >= self.stabilizer:
                return f"Bet ${0.01 * self.bankroll:0.2f} ({self.EV[1]:0.2f}, {self.game.w2:0.0%}) on {self.game.t2} (v {self.game.t1}) {'\uEA6C' if self.warning else ''} \uEAA5"
        return "\ueab8"

import os
from dotenv import load_dotenv
from math import prod, sqrt
from game import Game
from sport import Sport
from predictor import SportCode, loadSport
import numpy as np


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

    def analyzeParlays(self) -> None:
        self.parlays: list[Parlay] = []

        for i in range(len(self.analyses)):
            for j in range(i + 1, len(self.analyses)):
                self.parlays.append(
                    Parlay(
                        games=[self.analyses[i], self.analyses[j]],
                        sport=self.sport,
                        kelly_fraction=self.kelly_fraction,
                        bankroll=self.bankroll,
                    )
                )

        for i in range(len(self.analyses)):
            for j in range(i + 1, len(self.analyses)):
                for k in range(j + 1, len(self.analyses)):
                    self.parlays.append(
                        Parlay(
                            games=[
                                self.analyses[i],
                                self.analyses[j],
                                self.analyses[k],
                            ],
                            sport=self.sport,
                            kelly_fraction=self.kelly_fraction,
                            bankroll=self.bankroll,
                        )
                    )

        for i in range(len(self.analyses)):
            for j in range(i + 1, len(self.analyses)):
                for k in range(j + 1, len(self.analyses)):
                    for l in range(k + 1, len(self.analyses)):
                        self.parlays.append(
                            Parlay(
                                games=[
                                    self.analyses[i],
                                    self.analyses[j],
                                    self.analyses[k],
                                    self.analyses[l],
                                ],
                                sport=self.sport,
                                kelly_fraction=self.kelly_fraction,
                                bankroll=self.bankroll,
                            )
                        )

        for p in self.parlays:
            p.analyze()

        for p in sorted(self.parlays, key=lambda x: x.tev * sqrt(x.tp), reverse=True):
            print(p)


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

        return self

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


class Parlay:
    def __init__(
        self, games: list[Odds], sport: Sport, kelly_fraction: float, bankroll: float
    ) -> None:
        self.games = games
        self.sport = sport
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll

    def calcFStar(self):
        return self.tev / (self.tor - 1) * self.kelly_fraction

    def betAmount(self):
        return np.clip(self.f_star, 0, 0.05, dtype=float) * self.bankroll

    def analyze(self):
        tp = []
        tip = []
        tev = []
        tor = []
        self.pick = []
        self.skip = False

        for g in self.games:
            if g.EV[0] > 0:
                self.pick.append(0)
            elif g.EV[1] > 0:
                self.pick.append(1)
            else:
                self.skip = True
                continue
            tp.append(g.game.w1 if self.pick[-1] == 0 else g.game.w2)
            tip.append(g.ip[self.pick[-1]])
            tev.append(g.EV[self.pick[-1]])
            tor.append(g.OR[self.pick[-1]])
        if not self.skip:
            self.tp = prod(tp)
            self.tip = prod(tip)
            self.tev = sum(tev)
            self.tor = prod(tor)
            self.f_star = self.calcFStar()
        else:
            self.tp = 0.0
            self.tip = 0.0
            self.tev = 0.0
            self.tor = 0.0
            self.f_star = 0.0

    def __str__(self) -> str:
        if self.tev <= 1.0 or self.skip:
            return "\ueab8"
        picks = [
            g.game.t1 if pick == 0 else g.game.t2
            for g, pick in zip(self.games, self.pick)
        ]
        if self.f_star * self.bankroll >= 0.1:
            return f"Bet ${self.betAmount():0.2f} ({self.tev:0.2f}, {self.tp:0.0%}) {' + '.join(picks)}"
        return "\ueab8"

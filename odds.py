import os
from dotenv import load_dotenv
from math import prod
from game import Game
from sport import Sport
from predictor import SportCode, loadSport
import numpy as np
import scipy.stats as stats


class BettingEngine:
    def __init__(self, sportCode: SportCode, year: int) -> None:
        self.sport = loadSport(sportCode, year)
        self.loadConfig()
        self.loadOdds()
        self.analyze()
        self.analyzeParlays()
        self.recommend()

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
                    float(spread),
                    (int(spreadOdds1), int(spreadOdds2)),
                    sport=self.sport,
                    kelly_fraction=self.kelly_fraction,
                    bankroll=self.bankroll,
                    stabilizer=self.stabilizer,
                )
                for t1, t2, o1, o2, spread, _, spreadOdds1, spreadOdds2 in odds
            ]

        self.odds = odds

    def analyze(self) -> None:
        self.analyses = [odds.analyze() for odds in self.odds]

    def analyzeParlays(self) -> None:
        self.parlays: list[Parlay] = []

        for i in range(len(self.analyses)):
            if self.analyses[i].shouldSkip():
                continue
            for j in range(i + 1, len(self.analyses)):
                if self.analyses[j].shouldSkip():
                    continue
                self.parlays.append(
                    Parlay(
                        games=[self.analyses[i], self.analyses[j]],
                        sport=self.sport,
                        kelly_fraction=self.kelly_fraction,
                        bankroll=self.bankroll,
                    )
                )

        for i in range(len(self.analyses)):
            if self.analyses[i].shouldSkip():
                continue
            for j in range(i + 1, len(self.analyses)):
                if self.analyses[j].shouldSkip():
                    continue
                for k in range(j + 1, len(self.analyses)):
                    if self.analyses[k].shouldSkip():
                        continue
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
            if self.analyses[i].shouldSkip():
                continue
            for j in range(i + 1, len(self.analyses)):
                if self.analyses[j].shouldSkip():
                    continue
                for k in range(j + 1, len(self.analyses)):
                    if self.analyses[k].shouldSkip():
                        continue
                    for l in range(k + 1, len(self.analyses)):
                        if self.analyses[l].shouldSkip():
                            continue
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

    def recommend(self):
        target = 5

        four_leg_parlays = [
            p
            for p in self.parlays
            if len(p.games) == 4 and not getattr(p, "skip", False)
        ]
        best_parlay = max(
            (p for p in four_leg_parlays if len(str(p)) > 1),
            key=lambda p: getattr(p, "tp", 0.0),
            default=None,
        )
        if best_parlay and len(str(best_parlay)) > 1:
            print(best_parlay)

        sorted_analyses = sorted(
            self.analyses, key=lambda a: max(a.game.w1, a.game.w2), reverse=True
        )

        selected = []

        for a in sorted_analyses:
            if len(selected) >= target:
                break
            if not a.shouldSkip():
                selected.append(a)

        for a in sorted_analyses:
            if len(selected) >= target:
                break
            if a in selected:
                continue
            if not a.shouldSkip():
                selected.append(a)

        for s in selected:
            print(s)

        for o in sorted(self.odds, key=lambda o: max(o.spreadEV), reverse=True)[:10]:
            if o.spreadEV[0] > 0.2 or o.spreadEV[1] > 0.2:
                if o.spreadEV[0] > o.spreadEV[1]:
                    print(
                        f"Bet ${self.bankroll / 20:0.2f} on {o.t1} at {o.spread:+} ({o.spreadEV[0]:0.2f})"
                    )
                else:
                    print(
                        f"Bet ${self.bankroll / 20:0.2f} on {o.t2} at {-o.spread:+} ({o.spreadEV[1]:0.2f})"
                    )


class Odds:
    def __init__(
        self,
        t1: str,
        t2: str,
        o1: int,
        o2: int,
        spread: float,
        spreadOdds: tuple[int, int],
        sport: Sport,
        kelly_fraction: float,
        bankroll: float,
        stabilizer: float,
    ):
        self.t1 = t1
        self.t2 = t2
        self.o1 = o1
        self.o2 = o2
        self.spread = spread or 0.0
        self.spreadOdds = spreadOdds or (0, 0)
        self.sport = sport
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll
        self.stabilizer = stabilizer
        self.skip = False

    def calcIP(self) -> tuple[float, float]:
        return (
            -self.o1 / (100 - self.o1) if self.o1 < 0 else 100 / (self.o1 + 100),
            -self.o2 / (100 - self.o2) if self.o2 < 0 else 100 / (self.o2 + 100),
        )

    def calcDO(self) -> tuple[float, float]:
        # return decimal odds (DO)
        return (
            1 + (-100 / self.o1 if self.o1 < 0 else self.o1 / 100),
            1 + (-100 / self.o2 if self.o2 < 0 else self.o2 / 100),
        )

    def calcEV(self) -> tuple[float, float]:
        w1 = self.game.w1
        w2 = self.game.w2
        # EV per $1 stake using net odds `OR` (b): p*b - q
        return (w1 * self.OR[0] - w2, w2 * self.OR[1] - w1)

    def calcFStar(self) -> tuple[float, float]:
        # Use net odds (OR = DO - 1) as 'b' in Kelly formula
        return (
            (self.EV[0] / self.OR[0] * self.kelly_fraction) if self.OR[0] > 0 else 0.0,
            (self.EV[1] / self.OR[1] * self.kelly_fraction) if self.OR[1] > 0 else 0.0,
        )

    def calcSw(self) -> float:
        gap = abs(self.spread - self.game.spread1)
        Sw = float(stats.norm.cdf(gap / 11))
        return Sw

    def calcSpreadEV(self) -> tuple[float, float]:
        sor = (
            (
                100 / abs(self.spreadOdds[0])
                if self.spreadOdds[0] < 0
                else self.spreadOdds[0] / 100
            ),
            (
                100 / abs(self.spreadOdds[1])
                if self.spreadOdds[1] < 0
                else self.spreadOdds[1] / 100
            ),
        )
        return (
            self.Sw * sor[0] - (1 - self.Sw),
            self.Sw * sor[1] - (1 - self.Sw),
        )

    def pick(self):
        if self.EV[0] > self.EV[1]:
            return 0
        else:
            return 1

    def shouldSkip(self):
        if self.skip:
            return self.skip
        if self.EV[0] <= 0 and self.EV[1] <= 0:
            self.skip = True
            return self.skip
        if (
            self.EV[self.pick()] <= 0.2
            or self.OR[self.pick()] >= 4.00
            or self.OR[self.pick()] <= 1.25
        ):
            self.skip = True
            return self.skip
        if self.f_star[self.pick()] * self.bankroll < 0.1:
            self.skip = True
            return self.skip
        return False

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
        # calcDO returns decimal odds (DO); expose both DO and net-odds OR = DO - 1
        self.DO = self.calcDO()
        self.OR = (self.DO[0] - 1, self.DO[1] - 1)
        self.EV = self.calcEV()
        self.f_star = self.calcFStar()
        self.Sw = self.calcSw()
        self.spreadEV = self.calcSpreadEV()

        self.warning = (self.EV[0] > 0 and self.o1 > 0) or (
            self.EV[1] > 0 and self.o2 > 0
        )

        return self

    def betAmount(self) -> tuple[float, float]:
        return np.clip(self.f_star, 0, 0.05, dtype=float) * self.bankroll

    def __str__(self) -> str:
        if not self.shouldSkip():
            if self.pick() == 0:
                return f"Bet ${self.betAmount()[0]:0.2f} ({self.EV[0]:0.2f}, {self.game.w1:0.0%}) on {self.game.t1} (@ {self.game.t2}) {'\uEA6C' if self.warning else ''}"
            else:
                return f"Bet ${self.betAmount()[1]:0.2f} ({self.EV[1]:0.2f}, {self.game.w2:0.0%}) on {self.game.t2} (@ {self.game.t1}) {'\uEA6C' if self.warning else ''}"
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
        # use net parlay odds (OR = DO - 1)
        if getattr(self, "OR", 0) <= 0:
            return 0.0
        return self.tev / self.OR * self.kelly_fraction

    def betAmount(self):
        return np.clip(self.f_star, 0, 0.05, dtype=float) * self.bankroll

    def shouldSkip(self):
        if self.skip:
            return self.skip
        if self.tev <= 1.0 or self.f_star * self.bankroll >= 0.1:
            self.skip = True
            return self.skip

    def analyze(self):
        tp = []
        tip = []
        tev = []
        do_list = []
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
            # do_list is product of decimal odds (DO)
            do_list.append(g.DO[self.pick[-1]])
        if not self.skip:
            self.tp = prod(tp)
            self.tip = prod(tip)
            # parlay total probability (product of individual win probs)
            # DO: product of decimal odds; OR (net) = DO - 1
            self.DO = prod(do_list)
            self.OR = self.DO - 1
            self.tp = prod(tp)
            # parlay expected value for $1 stake: P_all * (DO - 1) - (1 - P_all) = P_all*DO - 1
            self.tev = (self.tp * self.DO) - 1
            self.f_star = self.calcFStar()
        else:
            self.tp = 0.0
            self.tip = 0.0
            self.tev = 0.0
            self.DO = 0.0
            self.OR = 0.0
            self.f_star = 0.0

    def __str__(self) -> str:
        if not self.shouldSkip():
            picks = [
                g.game.t1 if pick == 0 else g.game.t2
                for g, pick in zip(self.games, self.pick)
            ]
            return f"Bet ${self.betAmount():0.2f} ({self.tev:0.2f}, {self.tp:0.0%}) {' + '.join(picks)}"
        return "\ueab8"

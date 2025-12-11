from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional
import warnings
import math
from math import exp, floor, ceil, sqrt
import numpy as np

from utils import avg, gaussian, getroots, platt_scale, filepath, getcoeff

if TYPE_CHECKING:
    from sport import Sport


class Game:
    def __init__(
        self,
        h1: bool,
        t1: str,
        p1: Optional[int],
        h2: bool,
        t2: str,
        p2: Optional[int],
        sport: "Sport",
        postseason: Optional[int | str] = 0,
        day=datetime.strptime("2000-01-01", r"%Y-%m-%d"),
        id=None,
    ):
        """ """

        self.h1 = h1
        self.t1 = t1
        self.p1 = p1
        self.h2 = h2
        self.t2 = t2
        self.p2 = p2
        self.ps = postseason
        self.day = day
        self.id = id
        self.week = 0
        self.platt = (0.0, 0.0)
        self.sport = sport

        self.r1: int | Literal[""]
        self.r2: int | Literal[""]
        self.badness: float | Literal[""]

        self.pahaa: float
        self.pfhaa: float
        self.paoaa: float
        self.pfoaa: float

        self.p_flag = 0

        if self.p1 == 0 and self.p2 == 0:
            if self.sport.s != "iru":
                self.p1 = None
                self.p2 = None
            elif self.sport.s == "iru" and self.day > datetime(
                2015, 8, 8, 14, 6, 34, 358973
            ):
                self.p1 = None
                self.p2 = None

        if self.p1 is not None and self.p2 is not None:
            self.p_flag = 1

    def w(self):
        i1 = self.sport.teams.index(self.t1)
        i2 = self.sport.teams.index(self.t2)

        if (i1 is None or self.t1 is None) and not (i2 is None or self.t2 is None):
            if (
                self.sport.s == "fcs"
                and self.t1
                in open(f"{filepath}\\fbs\\{self.sport.year}\\teams.csv").read()
            ):
                self.w1 = 1
                self.w2 = 0
                self.spread1 = ""
                self.spread2 = ""
            else:
                self.w1 = 0
                self.w2 = 1
                self.spread1 = ""
                self.spread2 = ""
            self.v1 = self.wvar(self.p1, self.p2, self.w1)
            self.v2 = self.wvar(self.p2, self.p1, self.w2)
            return
        elif not (i1 is None or self.t1 is None) and (i2 is None or self.t2 is None):
            if (
                self.sport.s == "fcs"
                and self.t2
                in open(f"{filepath}\\fbs\\{self.sport.year}\\teams.csv").read()
            ):
                self.w1 = 0
                self.w2 = 1
                self.spread1 = ""
                self.spread2 = ""
            else:
                self.w1 = 1
                self.w2 = 0
                self.spread1 = ""
                self.spread2 = ""
            self.v1 = self.wvar(self.p1, self.p2, self.w1)
            self.v2 = self.wvar(self.p2, self.p1, self.w2)
            return
        elif (i1 is None or self.t1 is None) and (i2 is None or self.t2 is None):
            self.sport.log(self.day, ":", self.t1, "vs", self.t2, "is not a real game")

            try:
                self.sport.games.remove(self)
            except:
                self.w1 = 0.5
                self.w2 = 0.5
            return

        try:
            if (t1 := self.sport.teams[i1]) and (t2 := self.sport.teams[i2]):
                tpfh1 = t1.tpfh
                tpah1 = t1.tpah
                n1 = t1.n
                tpfh2 = t2.tpfh
                tpah2 = t2.tpah
                n2 = t2.n

                mo1, bo1, md1, bd1, mnd1, mxd1, mno1, mxo1 = (
                    t1.mo,
                    t1.bo,
                    t1.md,
                    t1.bd,
                    t1.mnd,
                    t1.mxd,
                    t1.mno,
                    t1.mxo,
                )
                mo2, bo2, md2, bd2, mnd2, mxd2, mno2, mxo2 = (
                    t2.mo,
                    t2.bo,
                    t2.md,
                    t2.bd,
                    t2.mnd,
                    t2.mxd,
                    t2.mno,
                    t2.mxo,
                )

                if not self.p_flag:
                    self.pfhaa = tpfh1 / n1
                    self.pahaa = tpah1 / n1
                    self.pfoaa = tpfh2 / n2
                    self.paoaa = tpah2 / n2
                else:
                    self.pfhaa = (tpfh1 - (self.p1 or 0)) / (n1 - 1)
                    self.pahaa = (tpah1 - (self.p2 or 0)) / (n1 - 1)
                    self.pfoaa = (tpfh2 - (self.p2 or 0)) / (n2 - 1)
                    self.paoaa = (tpah2 - (self.p1 or 0)) / (n2 - 1)

                (a, b, c) = getcoeff(
                    self.paoaa * (mo1 * self.paoaa + bo1 + mno1),
                    abs(mno1),
                    self.paoaa * (mo1 * self.paoaa + bo1),
                    1,
                    self.paoaa * (mo1 * self.paoaa + bo1 + mxo1),
                    abs(mxo1),
                )

                (d, e, f) = getcoeff(
                    self.pfhaa * (md2 * self.pfhaa + bd2 + mnd2),
                    abs(mnd2),
                    self.pfhaa * (md2 * self.pfhaa + bd2),
                    1,
                    self.pfhaa * (md2 * self.pfhaa + bd2 + mxd2),
                    abs(mxd2),
                )

                (self.phl, self.phh) = getroots(a, b, c, d, e, f)

                (a, b, c) = getcoeff(
                    self.pahaa * (mo2 * self.pahaa + bo2 + mno2),
                    abs(mno2),
                    self.pahaa * (mo2 * self.pahaa + bo2),
                    1,
                    self.pahaa * (mo2 * self.pahaa + bo2 + mxo2),
                    abs(mxo2),
                )

                (d, e, f) = getcoeff(
                    self.pfoaa * (md1 * self.pfoaa + bd1 + mnd1),
                    abs(mnd1),
                    self.pfoaa * (md1 * self.pfoaa + bd1),
                    1,
                    self.pfoaa * (md1 * self.pfoaa + bd1 + mxd1),
                    abs(mxd1),
                )

                (self.pal, self.pah) = getroots(a, b, c, d, e, f)

        except:
            self.sport.log(f"Could not get roots for {self.t1} vs {self.t2}")
            self.w1 = 0.5
            self.w2 = 0.5
            return

        t = []

        for i in range(self.phh - self.phl):
            t.append([])
            for j in range(self.pah - self.pal):
                if (self.phl + i) > (self.pal + j):
                    t[i].append(1)
                elif (self.phl + i) == (self.pal + j):
                    t[i].append(0.5)
                elif (self.phl + i) < (self.pal + j):
                    t[i].append(0)

        for i in range(self.phh - self.phl):
            t[i] = sum(t[i])

        s = sum(t)
        if self.phh == self.phl:
            self.phh += 1
        if self.pah == self.pal:
            self.pah += 1
        self.w1 = s / ((self.phh - self.phl) * (self.pah - self.pal))

        if self.h1 and not self.h2:
            if self.w1 + self.sport.ha < 1:
                self.w1 = self.w1 + self.sport.ha
            else:
                self.w1 = 1
        elif self.h2 and not self.h1:
            if self.w1 - self.sport.ha < 0:
                self.w1 = 0
            else:
                self.w1 = self.w1 - self.sport.ha

        if self.sport.platt[0] != 0:
            self.platt = self.sport.platt
            self.w1 = float(
                platt_scale(self.w1, self.sport.platt[0], self.sport.platt[1])
            )
            self.w2 = float(
                platt_scale(1 - self.w1, self.sport.platt[0], self.sport.platt[1])
            )
        elif self.platt[0] != 0:
            self.w1 = float(platt_scale(self.w1, self.platt[0], self.platt[1]))
            self.w2 = float(platt_scale(1 - self.w1, self.platt[0], self.platt[1]))
        else:
            self.w2 = 1 - self.w1

        self.proj1 = int(avg([self.phh, self.phl]))
        self.proj2 = int(avg([self.pah, self.pal]))

        if self.p1 is not None and self.p2 is not None:
            self.v1 = self.wvar(self.p1, self.p2, self.w1)
            self.v2 = self.wvar(self.p2, self.p1, self.w2)
            if self.w1 > 0.5:
                if self.p1 > self.p2:
                    self.brier = (1 - self.w1) ** 2
                else:
                    self.brier = (0 - self.w1) ** 2
            else:
                if self.p1 <= self.p2:
                    self.brier = (1 - self.w2) ** 2
                else:
                    self.brier = (0 - self.w2) ** 2
        else:
            self.v1 = self.wvar(self.proj1, self.proj2, self.w1)
            self.v2 = self.wvar(self.proj2, self.proj1, self.w2)

        self.spread1 = avg([self.pah, self.pal]) - avg([self.phl, self.phh])
        self.spread2 = -self.spread1
        self.ou = int(avg([self.pah, self.pal]) + avg([self.phl, self.phh]))

    def wvar(self, f, a, w):
        if f is None or a is None:
            return None

        i1 = self.sport.teams[self.t1]
        i2 = self.sport.teams[self.t2]

        m = f - a

        if f == a:
            self.v2 = float(0)
            self.v1 = float(0)
            return float(0)

        if m < 0:
            z = (
                self.sport.mov
                * math.tanh(m / self.sport.mov)
                * 0.5
                * math.exp(1.386 * w)
            )
        elif m > 0:
            z = (
                self.sport.mov
                * math.tanh(m / self.sport.mov)
                * 2
                * math.exp(-1.386 * w)
            )

        if self.ps:
            if m < 0:  # if loss
                z = min(
                    z
                    + self.sport.pweight(
                        self.t1, self.p1 or 0, self.t2, self.p2 or 0, self.sport.s
                    ),
                    -1.000,
                )
            elif m > 0:  # if win
                z += self.sport.pweight(
                    self.t1, self.p1 or 0, self.t2, self.p2 or 0, self.sport.s
                )

        if self.sport.s in ["fbs", "mbb", "wbb"] and (not i1 or not i2):
            if m < 0:
                z *= 2
            elif m > 0:
                z /= 2

        return z

    def display(self):
        try:
            print(
                self.r1,
                self.t1,
                self.proj1,
                self.proj2,
                self.t2,
                self.r2,
                self.w1,
                self.spread1,
                self.ou,
                self.v1,
                sep=",",
            )
        except:
            print(self.r1, self.t1, "", "", self.t2, self.r2, "", "", "", "", sep=",")


class Games(list[Game]):
    pass

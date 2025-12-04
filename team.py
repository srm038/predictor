from typing import TYPE_CHECKING, Literal, Optional
import warnings
import math
from math import copysign as sign
from math import floor, ceil
from game import Game
from scipy import stats
import statsmodels.nonparametric.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from utils import avg

if TYPE_CHECKING:
    from sport import Sport


class Team:
    def __init__(self, codename, name, sport: "Sport", conf="INDY", **kwargs):
        """ """

        self.codename = codename
        self.name = name
        self.pfh = []
        self.pah = []
        self.sched = []
        self.conference = conf
        self.sport = sport
        self.color = "black"

        self.rank: int
        self.wrank: int

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, value: str) -> bool:
        return self.codename == value

    def updatestats(self):
        self.sched = []
        self.pah = []
        self.pfh = []

        for i in self.sport.games:
            if i.t1 == self.codename or i.t2 == self.codename:
                self.sched.append(i)

        self.hindex()

        self.oppindex = []

        for i in self.sched:
            if i.t1 == self.codename:
                self.pfh.append(i.p1)
                self.pah.append(i.p2)
                self.oppindex.append(self.sport.teams.index(i.t2))
            else:
                self.pfh.append(i.p2)
                self.pah.append(i.p1)
                self.oppindex.append(self.sport.teams.index(i.t1))
        if self.sport.s == "iru" and len(self.pfh) > 50:
            self.pfh = self.pfh[-50:]
            self.pah = self.pah[-50:]

        self.n = len([x for x in self.pfh if x is not None])
        self.tpfh = sum([x for x in self.pfh if x is not None])
        self.tpah = sum([x for x in self.pah if x is not None])

    def updatemetrics(self):
        temp_pfh = [list(self.pfh), [], []]
        temp_pah = [list(self.pah), [], []]

        for i in range(len(self.sched)):
            if self.sched[i].t1 == self.codename:
                self.oppindex[i] = self.sport.teams.index(self.sched[i].t2)
            else:
                self.oppindex[i] = self.sport.teams.index(self.sched[i].t1)

        for i in self.oppindex[: self.n]:
            temp_pfh[1].append(None)
            if i is None:
                temp_pfh[2].append(None)
            else:
                temp_pfh[2].append(
                    [self.sport.teams[i].tpah, self.sport.teams[i].n - 1]
                )

        for i in range(self.n):
            if temp_pfh[0][i] is None:
                temp_pfh[1][i] = None
                temp_pfh[2][i] = None
            elif temp_pfh[2][i] is None:
                temp_pfh[0][i] = None
                temp_pfh[1][i] = None
            else:
                try:
                    temp_pfh[1][i] = (temp_pfh[2][i][0] - temp_pfh[0][i]) / temp_pfh[2][
                        i
                    ][1]
                except ZeroDivisionError:
                    self.sport.log(
                        self.name + " plays a team with only one division game"
                    )
                    return
                try:
                    temp_pfh[0][i] = temp_pfh[0][i] / temp_pfh[1][i]
                except ZeroDivisionError:
                    self.sport.log(
                        self.name + "plays a team with only one division game"
                    )
                    # who is this?
                    return
                except TypeError:
                    self.sport.log(self.name + " doesnt play any division games")
                    self.sport.log(self.name + " plays " + self.n + " games")
                    return

        for i in self.oppindex[: self.n]:
            temp_pah[1].append(None)
            if i is None:
                temp_pah[2].append(None)
            else:
                temp_pah[2].append(
                    [self.sport.teams[i].tpfh, self.sport.teams[i].n - 1]
                )

        for i in range(self.n):
            if temp_pah[0][i] is None:
                temp_pah[1][i] = None
                temp_pah[2][i] = None
            elif temp_pah[2][i] is None:
                temp_pah[0][i] = None
                temp_pah[1][i] = None
            else:
                try:
                    temp_pah[1][i] = (temp_pah[2][i][0] - temp_pah[0][i]) / temp_pah[2][
                        i
                    ][1]
                except ZeroDivisionError:
                    self.sport.log(self.codename, " plays a team with only one game")
                    return
                try:
                    temp_pah[0][i] = temp_pah[0][i] / temp_pah[1][i]
                except ZeroDivisionError:
                    self.sport.log(self.codename, " plays a team with only one game")
                    return
                except TypeError:
                    self.sport.log(self.codename, " doesnt play any division games")
                    return

        temp_pfh[0] = [x for x in temp_pfh[0] if x is not None]
        temp_pfh[1] = [x for x in temp_pfh[1] if x is not None]
        temp_pfh[2] = [x for x in temp_pfh[2] if x is not None]
        temp_pah[0] = [x for x in temp_pah[0] if x is not None]
        temp_pah[1] = [x for x in temp_pah[1] if x is not None]
        temp_pah[2] = [x for x in temp_pah[2] if x is not None]

        if (
            len(temp_pfh[0]) == 0
            or len(temp_pfh[1]) == 0
            or len(temp_pfh[2]) == 0
            or len(temp_pah[0]) == 0
            or len(temp_pah[1]) == 0
            or len(temp_pah[2]) == 0
        ):
            self.sport.log(self.name, " has not played any division games this season")
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitO = np.polyfit(temp_pfh[1], temp_pfh[0], 1)
                fitD = np.polyfit(temp_pah[1], temp_pah[0], 1)
            except ValueError:
                self.sport.log(
                    "{:}, {:}, {:}, {:}, {:}, {:}, {:}, {:}".format(
                        self.sched[self.pfh.index(None)].id,
                        self.codename,
                        self.sched[self.pfh.index(None)].t2,
                        self.pfh,
                        temp_pfh[1],
                        temp_pfh[0],
                        self.pah,
                        temp_pah[1],
                        temp_pah[0],
                    )
                )
                return

        self.mo = 0 if math.isnan(fitO[0]) else float(fitO[0])
        self.bo = 0 if math.isnan(fitO[1]) else float(fitO[1])
        self.md = 0 if math.isnan(fitD[0]) else float(fitD[0])
        self.bd = 0 if math.isnan(fitD[1]) else float(fitD[1])
        self.oeff = self.bo - 300 * self.mo**2
        self.deff = -self.bd - 360 * self.md**2

        self.mnd = []
        self.mxd = []
        self.mno = []
        self.mxo = []

        for i in range(len(temp_pfh[0])):
            e = temp_pfh[0][i] - (self.mo * temp_pfh[1][i] + self.bo)
            if e < 0:
                self.mno.append(e)
            elif e > 0:
                self.mxo.append(e)

        for i in range(len(temp_pah[0])):
            e = temp_pah[0][i] - (self.md * temp_pah[1][i] + self.bd)
            if e < 0:
                self.mnd.append(e)
            elif e > 0:
                self.mxd.append(e)

        for i in [self.mnd, self.mxd, self.mno, self.mxo]:
            if len(i) == 0:
                if i == self.mnd:
                    self.mnd = [self.bd - min(temp_pah[0])]
                elif i == self.mxd:
                    self.mxd = [self.bd + max(temp_pah[0])]
                elif i == self.mno:
                    self.mno = [self.bo - min(temp_pfh[0])]
                elif i == self.mxo:
                    self.mxo = [self.bo + max(temp_pfh[0])]
                print(self.name)

        self.mnd = sign(avg(self.mnd), -1)
        self.mxd = sign(avg(self.mxd), 1)
        self.mno = sign(avg(self.mno), -1)
        self.mxo = sign(avg(self.mxo), 1)

        if self.mnd < -1:
            self.mnd = -0.9
        if self.mxd > 1:
            self.mxd = 0.9
        if self.mno < -1:
            self.mno = -0.9
        if self.mxo > 1:
            self.mxo = 0.9

    def ytd(self):
        self.sport.log("Year to date results for " + self.codename)
        self.sport.log("Calculating LOESS for " + self.codename)
        try:
            self.smoothed()
        except ZeroDivisionError:
            self.sport.log("Cant compute LOESS for " + self.codename)
            self.loess = ["" for i in range(self.n)]

        i = 0
        for g in self.sched:
            try:
                proj1 = g.proj1
            except AttributeError:
                proj1 = ""
            try:
                proj2 = g.proj2
            except AttributeError:
                proj2 = ""
            try:
                spread1 = g.spread1
            except AttributeError:
                spread1 = ""
            try:
                spread2 = g.spread2
            except AttributeError:
                spread2 = ""
            try:
                w1 = g.w1
            except AttributeError:
                w1 = ""
            try:
                w2 = g.w2
            except AttributeError:
                w2 = ""
            if g.t1 == self.codename:
                if g.p_flag:
                    print(
                        "",
                        g.t2,
                        "",
                        g.p1,
                        g.p2,
                        w1,
                        spread1,
                        "",
                        g.v1,
                        self.loess[i],
                        sep=",",
                    )
                else:
                    print(
                        g.r2, g.t2, "", proj1, proj2, w1, spread1, "", "", "", sep=","
                    )
            else:
                if g.p_flag:
                    print(
                        "",
                        g.t1,
                        "",
                        g.p2,
                        g.p1,
                        w2,
                        spread2,
                        "",
                        g.v2,
                        self.loess[i],
                        sep=",",
                    )
                else:
                    print(
                        g.r1, g.t1, "", proj2, proj1, w2, spread2, "", "", "", sep=","
                    )
            i += 1

    def smoothed(self):
        a = []
        b = []
        o = []
        i = 1

        if self.n / 4 < 5:
            q = min(1, 5 / self.n)
        else:
            q = 1 / 4

        for s in self.sched:
            if not s.p_flag:
                continue
            a.append(i)
            if s.t1 == self.codename:
                b.append(s.v1)
                o.append(s.t2)
            else:
                b.append(s.v2)
                o.append(s.t1)
            i += 1

        p = sm.lowess(b, a, frac=q, return_sorted=False)

        self.loess = list(p)

        self.momentum = self.loess[-1]

    def wl(self, flag):
        """Return number of wins and losses"""

        win = 0
        loss = 0
        tie = 0
        win_perc = 0

        for i in self.sched:
            if i.p1 is None or i.p2 is None:
                continue
            elif i.t1 == self.codename:
                if i.p1 > i.p2:
                    win += 1
                elif i.p1 < i.p2:
                    loss += 1
                elif i.p1 == i.p1:
                    tie += 1
                win_perc += i.w1
            else:
                if i.p1 < i.p2:
                    win += 1
                elif i.p1 > i.p2:
                    loss += 1
                elif i.p1 == i.p1:
                    tie += 1
                win_perc += i.w2

        self.w = win
        self.l = loss
        self.t = tie

        if flag == 1:
            if tie:
                return str(win) + "-" + str(loss) + "-" + str(tie)
            else:
                return str(win) + "-" + str(loss)
        elif flag == 0:
            if tie:
                return "{:.3f}".format((win + tie / 2) / (win + loss + tie))
            else:
                return "{:.3f}".format(win / (win + loss))
        elif flag == 2:
            return str(win_perc) + "-" + str(len(self.sched) - win_perc)

    def wvara(self):
        a = []
        c = []

        for i in self.sched:
            if i.t1 == self.codename:
                try:
                    a.append(i.wvar(i.p1, i.p2, i.w1))
                except AttributeError:
                    a.append(i.wvar(i.p1, i.p2, 0.50))
                    i.w1 = 0.50
                    i.w2 = 0.50
                    i.v1 = i.wvar(i.p1, i.p2, 0.50)
                    i.v2 = i.wvar(i.p2, i.p1, 0.50)
            else:
                try:
                    a.append(i.wvar(i.p2, i.p1, i.w2))
                except AttributeError:
                    a.append(i.wvar(i.p2, i.p1, 0.50))
                    i.w2 = 0.50
                    i.w1 = 0.50
                    i.v1 = i.wvar(i.p1, i.p2, 0.50)
                    i.v2 = i.wvar(i.p2, i.p1, 0.50)

        for g in self.sched:
            if g.t1 == self.codename:
                if self.sport.teams[g.t2]:
                    if self.conference == self.sport.teams[g.t2].conference:
                        try:
                            c.append(g.wvar(g.p1, g.p2, g.w1))
                        except AttributeError:
                            c.append(g.wvar(g.p1, g.p2, 0.50))
            else:
                if self.sport.teams[g.t1]:
                    if self.conference == self.sport.teams[g.t1].conference:
                        try:
                            c.append(g.wvar(g.p2, g.p1, g.w2))
                        except AttributeError:
                            c.append(g.wvar(g.p2, g.p1, 0.50))

        self.cwvara = avg(c)

        return avg(a)

    def projwvara(self):
        """
        The average projected ranking of games yet to be played
        """

        a = []

        for g in self.sched:
            if not g.p_flag:
                if g.t1 == self.codename:
                    try:
                        a.append(g.v1)
                    except:
                        pass
                else:
                    try:
                        a.append(g.v2)
                    except:
                        pass

        return avg(a)

    def pyth(self):

        self.pythwin = self.tpfh**self.sport.pyth / (
            self.tpfh**self.sport.pyth + self.tpah**self.sport.pyth
        )

        return self.pythwin

    def graph(self):
        self.wvars = []
        fig = plt.figure()
        plt.cla()

        for g in self.sched:
            if g.t1 == self.name or g.t1 == self.codename:
                if g.p_flag:
                    self.wvars.append(g.v1)
                else:
                    self.wvars.append(0)
            else:
                if g.p_flag:
                    self.wvars.append(g.v2)
                else:
                    self.wvars.append(0)

        try:
            c = self.color
        except:
            c = "black"

        plt.bar(
            left=range(len(self.wvars)),
            height=self.wvars,
            width=0.8,
            color=c,
            alpha=0.5,
            edgecolor="none",
            align="center",
            linewidth=0,
        )

        if self.sport.s == "mbb":
            for g in range(
                ceil(min(self.wvars) / 5) * 5, floor(max(self.wvars) / 5 + 1) * 5, 5
            ):
                plt.plot([-0.5, self.n - 0.5], [g, g], color="white")
        else:
            for g in range(
                ceil(min(self.wvars) / 10) * 10,
                floor(max(self.wvars) / 10 + 1) * 10,
                10,
            ):
                plt.plot([-0.5, self.n - 0.5], [g, g], color="white")

        plt.plot([-0.5, self.n - 0.5], [0, 0], color="black")

        plt.plot(
            [-0.5, self.n - 0.5],
            [self.wvara(), self.wvara()],
            color="black",
            linestyle=":",
            linewidth=0.5,
        )

        fig.set_size_inches(4, 3)
        plt.axis("off")
        plt.xlim((-0.5, self.n))
        plt.title(
            self.codename,
            loc="left",
            family="Roboto",
            weight="black",
            y=0.95,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            alpha=0.75,
        )

        # plt.savefig(
        #     filepath
        #     + sport.sfile
        #     + "\\"
        #     + sport.year
        #     + "\\Plots\\"
        #     + self.codename
        #     + ".png"
        # )
        plt.close()

    def proj(self):
        w = 0
        l = 0
        t = 0

        for s in self.sched:
            if not s.p1 and not s.p2:
                if s.t1 == self.codename:
                    if s.w1 > 0.5:
                        w += 1
                    elif s.w1 == 0.5:
                        t += 1
                    else:
                        l += 1
                else:
                    if s.w2 > 0.5:
                        w += 1
                    elif s.w2 == 0.5:
                        t += 1
                    else:
                        l += 1

        self.projw = w + self.w
        self.projl = l + self.l
        self.projt = t + self.t

        if self.projt != 0:
            return "{:}-{:}-{:}".format(self.projw, self.projl, self.projt)
        else:
            return "{:}-{:}".format(self.projw, self.projl)

    def sos(self, t=0):

        r, soo0, soo1, sow0, sow1 = [], [], [], [], []

        if t not in ["soo", "sow", 0]:
            return "Type not defined"

        h = self.sport.firstFullWeek or self.sport.findFirstFullWeek()

        for s in self.sched:
            w = s.week
            if s.t1 == self.codename:
                b = self.sport.teams[s.t2]
                try:
                    win = s.w1
                except AttributeError:
                    print(s.t1, s.t2)
                    win = 0.5
            else:
                b = self.sport.teams[s.t1]
                try:
                    win = s.w2
                except AttributeError:
                    print(s.t1, s.t2)
                    win = 0.5
            if s.p_flag:
                sow0.append(win)
                if b is not None:
                    if w > h:
                        try:
                            soo0.append(self.sport.allwranks[b][w + 1])
                        except IndexError:
                            soo0.append(
                                next(
                                    x
                                    for x in reversed(self.sport.allwranks[b])
                                    if x is not None
                                )
                            )
                    else:
                        soo0.append(self.sport.allwranks[b][h + 1])
                else:
                    soo0.append(int(2 * len(self.sport.teams)))
            else:
                sow1.append(win)
                if b is not None:
                    if w > h:
                        try:
                            soo1.append(self.sport.allwranks[b][w + 1])
                        except IndexError:
                            soo1.append(
                                next(
                                    x
                                    for x in reversed(self.sport.allwranks[b])
                                    if x is not None
                                )
                            )
                    else:
                        soo1.append(self.sport.allwranks[b][h + 1])
                else:
                    soo1.append(int(2 * len(self.sport.teams)))
        if avg(sow1) == 0:
            sow1 = [1]
        if avg(soo1) == 0:
            soo1 = [len(self.sport.teams)]

        self.sow0 = avg(sow0) * 100
        self.sow1 = avg(sow1) * 100
        self.soo0 = (1 - avg(soo0) / len(self.sport.teams)) * 100
        self.soo1 = (1 - avg(soo1) / len(self.sport.teams)) * 100
        if t == 0:
            return

        if t == "sow":
            return "{:.2f},{:.2f}".format(self.sow0, self.sow1)
        elif t == "soo":
            return "{:.2f},{:.2f}".format(self.soo0, self.soo1)

    def power(self, p):
        """
        Returns list of teams that the home team has a p% chance of beating
        """

        p /= 100
        wins: list[int | float] = [0 for _ in range(len(self.sport.teams))]
        played = []

        for s in self.sched:
            if s.p1 and s.p2:
                if s.t1 == self.codename:
                    if s.t2 in played:
                        continue
                    else:
                        if s.p1 > s.p2:
                            if self.sport.teams[s.t2] is None:
                                continue
                            else:
                                wins[self.sport.teams.index(s.t2)] = 1
                        elif s.p1 < s.p2:
                            if self.sport.teams[s.t2] is None:
                                continue
                            else:
                                wins[self.sport.teams.index(s.t2)] = 0
                        played.append(s.t2)
                else:
                    if s.t1 in played:
                        continue
                    else:
                        if s.p1 < s.p2:
                            if self.sport.teams[s.t1] is None:
                                continue
                            else:
                                wins[self.sport.teams.index(s.t1)] = 1
                        elif s.p1 > s.p2:
                            if self.sport.teams[s.t1] is None:
                                continue
                            else:
                                wins[self.sport.teams.index(s.t1)] = 0
                        played.append(s.t1)

        for t in self.sport.teams:
            if t.codename == self.codename:
                continue
            elif t.codename in played:
                continue
            else:
                g = Game(
                    0,
                    self.codename,
                    None,
                    0,
                    t.codename,
                    None,
                    0,
                    datetime.strptime("2000-01-01", "%Y-%m-%d"),
                )
                try:
                    g.w()
                    if g.w1 < p:
                        wins[self.sport.teams.index(t.codename)] = 0
                    else:
                        wins[self.sport.teams.index(t.codename)] = 1
                except ZeroDivisionError:
                    wins[self.sport.teams.index(t.codename)] = 0.5
        return wins

    def power2(self):
        """
        Returns list of home teams w% against all others
        """

        wins: list[int | float] = [0 for _ in range(len(self.sport.teams))]
        played = []

        for s in self.sched:
            if s.p1 and s.p2:
                if s.t1 == self.codename:
                    if s.t2 in played:
                        continue
                    else:
                        if s.p1 > s.p2:
                            if self.sport.teams[s.t2] is None:
                                continue
                            else:
                                wins[self.sport.teams.index(s.t2)] = 1
                        elif s.p1 < s.p2:
                            if self.sport.teams[s.t2] is None:
                                continue
                            else:
                                wins[self.sport.teams.index(s.t2)] = 0
                        played.append(s.t2)
                else:
                    if s.t1 in played:
                        continue
                    else:
                        if s.p1 < s.p2:
                            if self.sport.teams[s.t1] is None:
                                continue
                            else:
                                wins[self.sport.teams.index(s.t1)] = 1
                        elif s.p1 > s.p2:
                            if self.sport.teams[s.t1] is None:
                                continue
                            else:
                                wins[self.sport.teams.index(s.t1)] = 0
                        played.append(s.t1)

        for t in self.sport.teams:
            if t.codename == self.codename:
                continue
            elif t.codename in played:
                continue
            else:
                g = Game(
                    0,
                    self.codename,
                    None,
                    0,
                    t.codename,
                    None,
                    0,
                    datetime.strptime("2000-01-01", "%Y-%m-%d"),
                )
                try:
                    g.w()
                    wins[self.sport.teams.index(t.codename)] = g.w1
                except ZeroDivisionError:
                    continue

        return wins

    def hindex(self):
        """
        Finds the largest number h such that the team has h wins with a margin of victory of at least h
        """

        movs = []

        for g in self.sched:
            if g.p_flag:
                if g.t1 == self.codename:
                    movs.append(g.p1 - g.p2)
                else:
                    movs.append(g.p2 - g.p1)

        if len(movs) == 0:
            self.H = 0
            self.sport.log("Couldnt compute h-index for " + self.codename)
            return

        counts = [len([m for m in movs if m >= i]) for i in range(1, max(movs) + 1)]

        for c, mov in enumerate(counts):
            if c + 1 <= mov:
                self.H = c + 1
        try:
            self.H
        except AttributeError:
            self.H = 0


class Teams(list):
    """List-like container for Team objects that supports lookup by
    team codename or full name using string keys, e.g. `teams['A']`.

    Behaves like a normal list for integer indices and slicing.
    """

    def __getitem__(self, key) -> Optional[Team]:
        if isinstance(key, str):
            for t in super().__iter__():
                try:
                    if t.codename == key or t.name == key:
                        return t
                except AttributeError:
                    continue
            return None
        return super().__getitem__(key)

    def index(self, value: str) -> Optional[int]:
        for i, t in enumerate(super().__iter__()):
            try:
                if t.codename == value or t.name == value:
                    return i
            except AttributeError:
                continue
        return None

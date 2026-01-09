import pickle
import math
from game import Game, Games
from team import Team, Teams
from utils import log, avg, brier, platt_scaling, filepath, Accuracy
from operator import itemgetter
import numpy as np
import csv
from datetime import datetime
from typing import Optional, TypedDict


class URLInfo(TypedDict):
    url: str
    sub: str


class Sport:
    def __init__(
        self,
        s,
        y,
        mov,
        ps,
        ha,
        pyth,
        url: URLInfo,
    ):
        """
        Initializes a new sport.

        Sport(s, y, mov, sfile, ps, ha, pyth)

        s: short code
        y: year
        mov: the margin of victory factor
        ps: postseason codes and factors
        ha: home advantage factor
        pyth: Pythagorean exponent
        """

        self.s = s
        self.mov = mov
        self.ps = ps
        self.year = y
        self.ha = ha
        self.pyth = pyth
        self.url = url
        self.platt = (0, 0)
        self.accuracy = 0.0
        self.rawAccuracy: list[Accuracy] = []
        self.brier = 1.0
        self.dayzero = datetime(int(y), 1, 1)

        self.teams: Teams = Teams()
        self.games: Games = Games()

        self.NR: list[tuple[str, int]] = []

        self.firstFullWeek: int
        self.maxNameLen: int
        self.fmeas: float

        self.dataraw = rf"{filepath}\{self.s}\{self.year}\raw.csv"
        self.teamsraw = rf"{filepath}\{self.s}\{self.year}\teams.csv"
        self.rankingraw = rf"{filepath}\{self.s}\{self.year}\rank.csv"
        self.accraw = rf"{filepath}\{self.s}\{self.year}\acc.csv"
        self.errraw = rf"{filepath}\{self.s}\{self.year}\err.csv"
        self.playoffraw = rf"{filepath}\{self.s}\{self.year}\playoff.csv"
        self.bayesconv = rf"{filepath}\{self.s}\{self.year}\bayes.csv"
        self.bayesrank = rf"{filepath}\{self.s}\{self.year}\bayesrank.csv"
        self.persistf = rf"{filepath}\{self.s}\{self.year}\persist.p"
        self.allranksf = rf"{filepath}\{self.s}\{self.year}\all.p"
        self.allwranksf = rf"{filepath}\{self.s}\{self.year}\all2.p"
        self.powf = rf"{filepath}\{self.s}\{self.year}\pow50.p"
        self.comprehensive = rf"{filepath}\{self.s}\{self.year}\allteams.csv"
        self.weekly = rf"{filepath}\{self.s}\{self.year}\weekly.csv"
        self.logfile = rf"{filepath}\{self.s}\{self.year}\log.txt"
        self.gamesfile = rf"{filepath}\{self.s}\{self.year}\games.csv"
        self.skinsfile = rf"{filepath}\{self.s}\{self.year}\skins.txt"

    def loadTeams(self):
        """Load teams from teamsraw file."""
        self.teams = Teams()

        with open(self.teamsraw, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                self.teams.append(Team(row[0], row[1], self, row[2]))

        for t in self.teams:
            t.skins = 1

    def loadGames(self):
        """Load games from dataraw file."""
        self.games = Games()
        open(self.gamesfile, "w").close()
        with open(self.dataraw, "r") as f:
            reader = csv.reader(f, delimiter=",")
            i = 1
            for row in reader:
                self.games.append(self.parseGame(row, i))
                i += 1
            del i

    def redistributeSkins(self):
        """Redistribute skins among teams based on games played."""
        self.log("Redistributing skins")
        for g in self.games:
            t1 = self.teams[g.t1]
            t2 = self.teams[g.t2]
            if t1 and t2:
                if g.p_flag and not g.ps and g.p1 and g.p2:
                    if g.p1 > g.p2:
                        t1.skins += t2.skins
                        t2.skins = 0
                    elif g.p2 > g.p1:
                        t2.skins += t1.skins
                        t1.skins = 0

        open(self.skinsfile, "w").close()
        with open(self.skinsfile, "a") as f:
            for t in self.teams:
                f.write(t.codename + "," + str(t.skins) + "\n")

    def log(self, *text):
        """Write log event to this sport's logfile (wrapper around utils.log)."""
        log(self.logfile, *text)

    def rankteams(self):

        self.loadGames()
        self.loadTeams()
        for t in self.teams:
            t.updatestats()
        for t in self.teams:
            t.updatemetrics()

        self.log("Ranking teams")

        #   WVARA

        self.log("Analyzing results")
        for g in self.games:
            g.w()

        wrank = [[], []]
        for t in self.teams:
            wrank[0].append(t.codename)
            wrank[1].append(t.wvara())
        temp = list(sorted(wrank[1], reverse=True))
        for t in self.teams:
            t.wrank = temp.index(t.wvara()) + 1

        for t in self.teams:
            soo0, sow0 = [], []
            soo1, sow1 = [], []
            for s in t.sched:
                if s.t1 == t.codename:
                    a = self.teams[s.t1]
                    b = self.teams[s.t2]
                    try:
                        win = s.w1
                    except AttributeError:
                        win = 0.5
                else:
                    b = self.teams[s.t1]
                    a = self.teams[s.t2]
                    try:
                        win = s.w2
                    except AttributeError:
                        win = 0.5
                if s.p_flag:
                    sow0.append(win)
                    if b is not None:
                        soo0.append(b.wrank)
                    else:
                        if self.s == "fcs" and (
                            s.t2
                            in open(rf"{filepath}\fbs\\{self.year}\\teams.csv").read()
                            or s.t1
                            in open(rf"{filepath}\fbs\\{self.year}\\teams.csv").read()
                        ):
                            soo0.append(-int(2 * len(self.teams)))
                        else:
                            soo0.append(int(2 * len(self.teams)))
                else:
                    sow1.append(win)
                    if b is not None:
                        soo1.append(b.wrank)
                    else:
                        if self.s == "fcs" and (
                            s.t2
                            in open(rf"{filepath}\fbs\\{self.year}\\teams.csv").read()
                            or s.t1
                            in open(rf"{filepath}\fbs\\{self.year}\\teams.csv").read()
                        ):
                            soo1.append(-int(2 * len(self.teams)))
                        else:
                            soo1.append(int(2 * len(self.teams)))
            t.sow0 = avg(sow0) * 100
            t.sow1 = avg(sow1) * 100
            t.soo0 = (1 - avg(soo0) / len(self.teams)) * 100
            t.soo1 = (1 - avg(soo1) / len(self.teams)) * 100
        sorank = [[], []]

        for i in self.teams:
            sorank[0].append(i.codename)
            sorank[1].append(i.soo0)

        temp = list(sorted(sorank[1], reverse=True))

        for i in self.teams:
            i.sorank = temp.index(i.soo0) + 1

        # SW-

        swrank = [[], []]

        for i in self.teams:
            swrank[0].append(i.codename)
            swrank[1].append(i.sow0)

        temp = list(sorted(swrank[1], reverse=True))

        for i in self.teams:
            i.swrank = temp.index(i.sow0) + 1

        rank = []

        for i in self.teams:
            rank.append(
                [i.codename, (11 * i.wrank + i.sorank + 3 * i.swrank) / 15, i.wrank]
            )

        temp = sorted(rank, key=itemgetter(1, 2), reverse=False)

        for i in self.teams:
            i.rank = (
                temp.index(
                    [i.codename, (11 * i.wrank + i.sorank + 3 * i.swrank) / 15, i.wrank]
                )
                + 1
            )

        for g in self.games:
            a = self.teams[g.t1]
            b = self.teams[g.t2]
            g.r1 = a.rank if a else ""
            g.r2 = b.rank if b else ""
            if a and b and g.r1 != "" and g.r2 != "":
                g.badness = g.r1 * g.r2 / len(self.teams) ** 2
            else:
                g.badness = ""

        self.log("Saving team rankings and stats")

        self.log("Calculating H-indices")

        for t in self.teams:
            t.hindex()

        open(self.rankingraw, "w").close()

        for t in sorted(self.teams, key=lambda i: i.rank or len(self.teams) + 1):
            with open(self.rankingraw, "a") as f:
                try:
                    if t.rank:
                        f.write(
                            f"{t.rank}\t{t.codename}\t{t.wvara():0.4f}\t{t.cwvara:0.4f}\t{t.wl(1)}\t"
                            f"{t.proj()}\t{t.sow0:0.0f}\t{t.sow1:0.0f}\t{t.soo0:0.0f}\t{
                            t.soo1:0.0f}\t{t.oeff:0.0f}\t{t.deff:0.0f}\t"
                            f"{t.H}\t{t.skins}\n"
                        )
                    else:
                        f.write(",".join(["", t.codename, "\n"]))
                except:
                    f.write(",".join(["", t.codename, "\n"]))
        for t in self.NR:
            with open(self.rankingraw, "a") as f:
                f.write(",".join(["", str(t), "\n"]))

        with open(self.persistf, "wb") as p:
            pickle.dump((self.teams, self.games), p)

    def rankwvara(self):
        print("Ranking teams by WVARA to date.....")

        pickle.dump((self.teams, self.games), open(self.persistf, "wb"))

        w = self.firstFullWeek or self.findFirstFullWeek()

        self.allwranks: list[list[Optional[int]]] = [
            [None for _ in range(1, self.games[-1].week + 2)] for _ in self.teams
        ]

        while w < self.currentweek:
            games = [g for g in self.games if g.week <= w]
            for t in self.teams:
                t.updatestats()
            for t in self.teams:
                t.updatemetrics()
            for g in games:
                try:
                    g.w()
                except ZeroDivisionError:
                    g.display()

            wrank = [[], []]

            for t in self.teams:
                wrank[0].append(t.codename)
                wrank[1].append(t.wvara())

            temp = list(sorted(wrank[1], reverse=True))

            for t, team in enumerate(self.teams):
                self.allwranks[t][w + 1] = temp.index(team.wvara()) + 1

            with open(self.persistf, "rb") as p:
                (self.teams, self.games) = pickle.load(p)
            w += 1

        with open(self.persistf, "rb") as p:
            (self.teams, self.games) = pickle.load(p)

        with open(self.allwranksf, "wb") as a:
            pickle.dump(self.allwranks, a)

        def update():

            c = self.currentweek - 1

            for t, team in enumerate(self.teams):
                try:
                    team.wrank = int(self.allwranks[t][-1] or len(self.teams) + 1)
                except TypeError:
                    team.wrank = int(self.allwranks[t][1:][c] or len(self.teams) + 1)
            for team in self.teams:
                team.sos()

            with open(self.persistf, "wb") as p:
                pickle.dump((self.teams, self.games), p)

        update()

    def rankconf(self):

        confs = []

        for t in self.teams:
            if t.conference in confs or t.conference == "INDY":
                continue
            else:
                confs.append(str(t.conference))

        for c in confs:
            cw = []
            q = []
            for t in self.teams:
                if t.conference == c:
                    cw.append(t.rank)
            q.append(min(cw))
            q.append(np.percentile(cw, 25))
            q.append(np.percentile(cw, 50))
            q.append(np.percentile(cw, 75))
            q.append(max(cw))
            print(q[0], q[1], q[2], q[3], q[4], c, avg(cw), sep="\t")

    def rankingtodate(self):

        try:
            if team := self.teams[0]:
                team.wrank
        except (AttributeError, UnboundLocalError):
            print("Couldnt find a rank.")

        self.log("Ranking teams by FULL to date.....")

        with open(self.persistf, "wb") as p:
            pickle.dump((self.teams, self.games), p)

        w = self.firstFullWeek or self.findFirstFullWeek()

        for g in self.games:
            if g.p_flag:
                self.currentweek = int(g.week)
            else:
                break

        self.allranks: list[list[Optional[int]]] = [
            [None for g in range(1, self.games[-1].week + 2)] for t in self.teams
        ]
        self.allwranks: list[list[Optional[int]]] = [
            [None for g in range(1, self.games[-1].week + 2)] for t in self.teams
        ]

        while w < self.currentweek:
            games = [g for g in self.games if g.week <= w]
            for t in self.teams:
                t.updatestats()
            for t in self.teams:
                t.updatemetrics()
            for g in games:
                try:
                    g.w()
                except ZeroDivisionError:
                    g.display()

            wrank = [[], []]

            for t in self.teams:
                wrank[0].append(t.codename)
                wrank[1].append(t.wvara())

            temp = list(sorted(wrank[1], reverse=True))

            for t, team in enumerate(self.teams):
                self.allwranks[t][w + 1] = temp.index(team.wvara()) + 1
                team.wrank = self.allwranks[t][w + 1] or len(self.teams) + 1
            for team in self.teams:
                team.sos()

            sorank = [[], []]

            for i in self.teams:
                sorank[0].append(i.codename)
                sorank[1].append(i.soo0)

            temp = list(sorted(sorank[1], reverse=True))

            for i in self.teams:
                i.sorank = temp.index(i.soo0) + 1

            swrank = [[], []]

            for i in self.teams:
                swrank[0].append(i.codename)
                swrank[1].append(i.sow0)

            temp = list(sorted(swrank[1], reverse=True))

            for i in self.teams:
                i.swrank = temp.index(i.sow0) + 1

            rank = []

            for t, team in enumerate(self.teams):
                wr = self.allwranks[t][w + 1]
                if wr is None:
                    self.log(f"Ranking error: {team.codename}")
                    return
                rank.append(
                    [
                        team.codename,
                        (11 * wr + 1 * team.sorank + 3 * team.swrank) / 15,
                        wr,
                    ]
                )

            temp = sorted(rank, key=itemgetter(1, 2), reverse=False)

            for t, team in enumerate(self.teams):
                wr = self.allwranks[t][w + 1]
                if wr is None:
                    self.log(f"Ranking error: {team.codename}")
                    return
                team.rank = (
                    temp.index(
                        [
                            team.codename,
                            (11 * wr + 1 * team.sorank + 3 * team.swrank) / 15,
                            wr,
                        ]
                    )
                    + 1
                )

            try:
                del wrank, sorank, swrank, temp
            except UnboundLocalError:
                pass

            try:
                for t, team in enumerate(self.teams):
                    self.allranks[t][w + 1] = team.rank
            except IndexError:
                pass

            (self.teams, self.games) = pickle.load(open(self.persistf, "rb"))
            w += 1

        for t in self.teams:
            t.updatestats()
        for t in self.teams:
            t.updatemetrics()
        for g in self.games:
            g.w()

        self.rankteams()

        for t, team in enumerate(self.teams):
            try:
                self.allranks[t].append(team.rank)
            except IndexError:
                pass

        self.log("{:} games total".format(len(self.games)))

        for g in self.games:
            t1 = self.teams.index(g.t1)
            t2 = self.teams.index(g.t2)
            if t1 is not None:
                try:
                    if r := self.allranks[t1][g.week + 1]:
                        g.r1 = r
                    else:
                        g.r1 = ""
                except IndexError:
                    g.r1 = ""
            else:
                g.r1 = ""
            if t2 is not None:
                try:
                    if r := self.allranks[t2][g.week + 1]:
                        g.r2 = r
                    else:
                        g.r2 = ""
                except IndexError:
                    g.r2 = ""
            else:
                g.r2 = ""

        self.printall()

        open(self.weekly, "w").close()

        tranposed = [
            [t[w] for t in self.allranks] for w in range(1, len(self.allranks[0]))
        ]

        with open(self.weekly, "w", encoding="utf-8") as f:
            i = 1
            teamlist = [t.name for t in self.teams]
            teamlist.insert(0, "week")
            f.write(",".join(teamlist) + "\n")
            for w in tranposed:
                w.insert(0, i)
                if w[2] == "":
                    continue
                else:
                    f.write(",".join([str(t) for t in w]) + "\n")
                i += 1

        with open(self.allranksf, "wb") as a:
            pickle.dump(self.allranks, a)
        with open(self.allwranksf, "wb") as a:
            pickle.dump(self.allwranks, a)
        with open(self.persistf, "wb") as a:
            pickle.dump((self.teams, self.games), a)

        self.log("Done")

    def conf(self, c="CUSA"):

        for t in self.teams:
            if t.conference == c:
                print(t.codename, t.cwvara, t.wvara(), sep="\t")

    def rankbays(self):
        m = []
        open(self.bayesconv, "w").close()

        print("Power.....")
        try:
            m = pickle.load(open(self.powf, "rb"))
        except IOError:
            for t in self.teams:
                m.append(list(t.power(50)))
            pickle.dump(m, open(self.powf, "wb"))
        print("Done")

        rank = []

        for i in m:
            if t := self.teams[m.index(i)]:
                rank.append((t.codename, avg(i)))

        rank = list(sorted(rank, key=itemgetter(1), reverse=True))

        def measure():

            k = 0
            s = 0
            for r in rank:
                k += 1
                for t in rank[k:]:
                    s += (
                        m[r0][t0]
                        if (r0 := self.teams.index(r[0]))
                        and (t0 := self.teams.index(t[0]))
                        else 0
                    )

            return s

        self.bayesperfect = len(self.teams) * (len(self.teams) - 1) / 2
        print(self.bayesperfect)

        def rearrange():

            amnt = round(float(4 * math.log10(len(self.teams))), 1)
            extra = 0

            first = round(float(measure()), 1)
            last = first - amnt

            for i in range(len(rank)):

                t = rank.pop(i)
                k = []
                r = 0

                while r < len(rank):
                    rank.insert(r, t)  # insert it
                    meas = measure()  # measure it
                    k.append(meas)  # record it
                    t = rank.pop(r)  # take it back out

                    if r == 0:
                        first = float(k[-1])
                        last = first - amnt

                    if i > 0 and k[-1] < last and r < amnt:
                        r += int(round(last - k[-1], 0))
                        print("skip", amnt, int(round(last - k[-1], 0)))

                    with open(self.bayesconv, "a") as b:
                        b.write(str(meas) + "\n")  # record it in file

                    if k[-1] <= (max(k) - amnt):
                        last = float(k[-1])
                        break  # check if its dropped too far
                    if k[-1] < last and r > amnt:
                        last = float(k[-1])
                        break
                    else:
                        last = float(k[-1])
                        r += 1  # check if it starts too far down

                last = max(k) - amnt
                k = k.index(max(k))

                rank.insert(k, t)

            fmeas = float(measure())

        rearrange()
        fmeaslast = self.fmeas - 1
        print(fmeaslast, self.fmeas, sep="\t")
        ranks = [None, list(rank)]

        if self.s == "nfl" or self.s == "nba":
            tol = 0.002
        else:
            tol = 0.0001

        while (1 - self.fmeas / fmeaslast) <= tol:
            ranks[0] = list(ranks[1])
            fmeaslast = float(self.fmeas)
            rearrange()
            print(fmeaslast, self.fmeas, sep="\t")
            ranks[1] = list(rank)
        else:
            rank = list(ranks[0])

        self.bayes = list(rank)
        self.bayesorder = self.fmeas / self.bayesperfect
        open(self.bayesrank, "w").close()

        print("{:.2%}".format(self.bayesorder))
        print(sum(1 for line in open(self.bayesconv)))
        for i in rank:
            print(*i, sep=",")
            with open(self.bayesrank, "a") as b:
                b.write(i[0] + "\n")

    def showweek(self, w):
        """ """

        self.log("Showing games for Week " + str(w))

        for g in self.games:
            if g.week == w and w > self.currentweek:
                if self.s != "nba" and self.s != "wbb" and self.s != "mbb":
                    try:
                        print(
                            g.r1,
                            g.t1,
                            g.proj1,
                            g.proj2,
                            g.t2,
                            g.r2,
                            g.w1,
                            g.spread1,
                            g.ou,
                            g.v1,
                            sep=",",
                        )
                    except:
                        print(g.r1, g.t1, "", "", g.t2, g.r2, "", "", "", "", sep=",")
                else:
                    try:
                        print(
                            g.r1,
                            g.t1,
                            g.proj1,
                            g.proj2,
                            g.t2,
                            g.r2,
                            g.w1,
                            g.spread1,
                            g.ou,
                            g.v1,
                            g.id,
                            sep=",",
                        )
                    except:
                        print(
                            g.r1,
                            g.t1,
                            "",
                            "",
                            g.t2,
                            g.r2,
                            "",
                            "",
                            "",
                            "",
                            g.id,
                            sep=",",
                        )
            elif g.week == w:
                if self.s != "nba" and self.s != "wbb" and self.s != "mbb":
                    print(g.t1, g.p1, sep=",")
                    print(g.t2, g.p2, sep=",")
                else:
                    print(g.id, g.t1, g.p1, sep=",")
                    print(g.id, g.t2, g.p2, sep=",")

    def showall(self):
        """ """

        self.log("Showing team rankings and stats")

        for t in self.teams:
            try:
                print(
                    t.rank,
                    t.codename,
                    t.wvara(),
                    t.cwvara,
                    t.wl(1),
                    t.proj(),
                    t.sow0,
                    t.sow1,
                    t.soo0,
                    t.soo1,
                    t.oeff,
                    t.deff,
                    t.H,
                    t.skins,
                    sep=",",
                )
            except:
                print("", t.codename, sep=",")
        for t in self.NR:
            print("", t, sep=",")

    def printall(self):

        self.log("Saving team and comprehensive files")

        open(self.comprehensive, "w").close()

        d = []
        c = 0
        maxNameLen = self.maxNameLen + 10
        s0 = "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}-{:}\t{:}\t{:.4f}"
        s1 = "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}-{:}\t{:}"
        div = "\n" + str("-" * 10) + "\n"

        for t in self.teams:
            teamfile = f"{filepath}\\{self.s}\\{self.year}\\Teams\\{t.codename}.txt"

            with open(teamfile, "w") as f:
                f.write("#{:} {:} ({:})".format(t.rank, t.name, t.wl(1)) + "\n")
                f.write("\n")
                f.write(
                    "Pyth: {:.3f} ({:.0f} wins)".format(
                        t.pyth(), t.pyth() * len(t.sched)
                    )
                    + "\n"
                )
                f.write("Proj: {:}".format(t.proj()) + "\n")
                f.write("\n")
                f.write("SoO: {:}".format(t.sos("soo")) + "\n")
                f.write("SoW: {:}".format(t.sos("sow")) + "\n")
                f.write("\n")
                try:
                    f.write("Oeff: {:.3f}".format(t.oeff) + "\n")
                    f.write("Deff: {:.3f}".format(t.deff) + "\n")
                except:
                    f.write("Oeff\n")
                    f.write("Deff\n")
                f.write("\n")

            with open(self.comprehensive, "a", encoding="utf-8") as f:
                f.write("#{:} {:} ({:})".format(t.rank, t.name, t.wl(1)) + "\n")
                f.write("\n")
                f.write(
                    "Pyth: {:.3f} ({:.0f} wins)".format(
                        t.pyth(), t.pyth() * len(t.sched)
                    )
                    + "\n"
                )
                f.write("Proj: {:}".format(t.proj()) + "\n")
                f.write("\n")
                f.write("SoO: {:}".format(t.sos("soo")) + "\n")
                f.write("SoW: {:}".format(t.sos("sow")) + "\n")
                f.write("\n")
                try:
                    f.write("Oeff: {:.3f}".format(t.oeff) + "\n")
                    f.write("Deff: {:.3f}".format(t.deff) + "\n")
                except:
                    f.write("Oeff\n")
                    f.write("Deff\n")
                f.write("\n")

            for g in t.sched:
                d.append([None for x in range(10)])
                if not g.p_flag:
                    p1 = "-"
                    p2 = "-"
                else:
                    p1 = g.p1
                    p2 = g.p2
                if g.t1 == t.codename:  # home is 1
                    if not self.teams[g.t2]:  # opponent is fcs
                        d[c][0] = g.day.strftime("%a %b %d")  # day
                        try:
                            d[c][1] = g.r1  # home rank
                        except:
                            d[c][1] = ""
                        if g.h1:
                            d[c][2] = ""  # location
                        elif not g.h1 and g.h2:
                            d[c][2] = "@"  # location
                        else:
                            d[c][2] = "·"  # location
                        d[c][3] = ""  # opponent rank
                        d[c][4] = g.t2  # opponent name
                        if p1 > p2:
                            d[c][5] = "W"  # win/loss
                        elif p1 < p2:
                            d[c][5] = "L"  # win/loss
                        else:
                            d[c][5] = "T"  # win/loss
                        if not g.p_flag:
                            d[c][5] = ""
                        d[c][6] = p1
                        d[c][7] = p2
                        d[c][8] = 1
                        try:
                            d[c][9] = g.v1
                        except AttributeError:
                            d[c][9] = 0
                    else:  # opponent is normal
                        d[c][0] = g.day.strftime("%a %b %d")  # day
                        try:
                            d[c][1] = g.r1  # home rank
                        except:
                            d[c][1] = ""
                        if g.h1:
                            d[c][2] = ""  # location
                        elif not g.h1 and g.h2:
                            d[c][2] = "@"  # location
                        else:
                            d[c][2] = "·"  # location
                        if not g.p_flag:  # opponent rank
                            if t2 := self.teams[g.t2]:
                                d[c][3] = t2.rank
                        else:
                            d[c][3] = g.r2  # opponent rank
                        if t2 := self.teams[g.t2]:
                            d[c][4] = t2.name  # opponent name
                        if p1 > p2:
                            d[c][5] = "W"  # win/loss
                        elif p1 < p2:
                            d[c][5] = "L"  # win/loss
                        else:
                            d[c][5] = "T"  # win/loss
                        if not g.p_flag:
                            d[c][5] = ""
                        d[c][6] = p1
                        d[c][7] = p2
                        try:
                            d[c][8] = g.w1
                        except AttributeError:
                            d[c][8] = "-"
                        try:
                            d[c][9] = g.v1
                        except AttributeError:
                            d[c][9] = 0
                else:  # home is 2
                    if not self.teams[g.t1]:  # opponent is fcs
                        d[c][0] = g.day.strftime("%a %b %d")  # day
                        try:
                            d[c][1] = g.r2  # home rank
                        except:
                            d[c][1] = ""
                        if g.h2:
                            d[c][2] = ""  # location
                        elif not g.h2 and g.h1:
                            d[c][2] = "@"  # location
                        else:
                            d[c][2] = "·"  # location
                        d[c][3] = ""  # opponent rank
                        d[c][4] = g.t1  # opponent name
                        if p1 < p2:
                            d[c][5] = "W"  # win/loss
                        elif p1 > p2:
                            d[c][5] = "L"  # win/loss
                        else:
                            d[c][5] = "T"  # win/loss
                        if not g.p_flag:
                            d[c][5] = ""
                        d[c][6] = p2
                        d[c][7] = p1
                        d[c][8] = 1
                        try:
                            d[c][9] = g.v2
                        except AttributeError:
                            d[c][9] = 0
                    else:  # opponent is normal
                        d[c][0] = g.day.strftime("%a %b %d")  # day
                        try:
                            d[c][1] = g.r2  # home rank
                        except:
                            d[c][1] = ""
                        if g.h2:
                            d[c][2] = ""  # location
                        elif not g.h2 and g.h1:
                            d[c][2] = "@"  # location
                        else:
                            d[c][2] = "·"  # location
                        if not g.p_flag:
                            if t1 := self.teams[g.t1]:
                                d[c][3] = t1.rank  # opponent rank
                        else:
                            d[c][3] = g.r1  # opponent rank
                        if t1 := self.teams[g.t1]:
                            d[c][4] = t1.name  # opponent name
                        if p1 < p2:
                            d[c][5] = "W"  # win/loss
                        elif p1 > p2:
                            d[c][5] = "L"  # win/loss
                        else:
                            d[c][5] = "T"  # win/loss
                        if not g.p_flag:
                            d[c][5] = ""
                        d[c][6] = p2
                        d[c][7] = p1
                        try:
                            d[c][8] = g.w2
                        except AttributeError:
                            d[c][8] = "-"
                        try:
                            d[c][9] = g.v2
                        except AttributeError:
                            d[c][9] = 0
                if d[c][9]:
                    with open(teamfile, "a", encoding="utf-8") as f:
                        f.write(s0.format(*d[c], max=maxNameLen) + "\n")
                    with open(self.comprehensive, "a", encoding="utf-8") as f:
                        f.write(s0.format(*d[c], max=maxNameLen) + "\n")
                else:
                    with open(teamfile, "a", encoding="utf-8") as f:
                        f.write(s1.format(*d[c], max=maxNameLen) + "\n")
                    with open(self.comprehensive, "a", encoding="utf-8") as f:
                        f.write(s1.format(*d[c], max=maxNameLen) + "\n")
                c += 1
            with open(teamfile, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("WVARA:  {:.4f}".format(t.wvara()) + "\n")
                f.write("CWVARA: {:.4f}".format(t.cwvara) + "\n")
                f.write(div)
            with open(self.comprehensive, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("WVARA:  {:.4f}".format(t.wvara()) + "\n")
                f.write("CWVARA: {:.4f}".format(t.cwvara) + "\n")
                f.write(div)

        for t in self.teams:
            t.hindex()

        open(self.rankingraw, "w").close()
        for i, teamI in enumerate(self.teams):
            for j, teamJ in enumerate(self.teams):
                if teamJ.rank == i + 1:
                    with open(self.rankingraw, "a", encoding="utf-8") as f:
                        try:
                            f.write(
                                ",".join(
                                    map(
                                        str,
                                        [
                                            teamJ.name,
                                            teamJ.wvara(),
                                            teamJ.cwvara,
                                            teamJ.wl(1) or "",
                                            teamJ.proj(),
                                            teamJ.sos("sow") or "",
                                            teamJ.sos("soo") or "",
                                            teamJ.oeff,
                                            (teamJ.deff),
                                            (teamJ.H),
                                            (teamJ.skins),
                                        ],
                                    )
                                )
                            )
                        except:
                            self.log("Error saving ranks")
                            pass

    def nextyear(self):
        """
        Take the data from the last year available and apply it to the coming year if possible
        """

        y1 = int(self.year)
        y2 = str(y1 + 1)

        dataraw0 = f"{filepath}\\{self.s}\\{y2}\\raw.csv"

        games = []

        with open(dataraw0, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if row[3] != "" and row[6] != "":
                    if "P" in row[7]:
                        if row[8] != "":
                            games.append(
                                Game(
                                    bool(int(row[1])),
                                    row[2],
                                    int(row[3]),
                                    bool(int(row[4])),
                                    row[5],
                                    int(row[6]),
                                    self,
                                    row[8],
                                )
                            )
                        else:
                            games.append(
                                Game(
                                    bool(int(row[1])),
                                    row[2],
                                    int(row[3]),
                                    bool(int(row[4])),
                                    row[5],
                                    int(row[6]),
                                    self,
                                    " ",
                                )
                            )
                    else:
                        games.append(
                            Game(
                                bool(int(row[1])),
                                row[2],
                                int(row[3]),
                                bool(int(row[4])),
                                row[5],
                                int(row[6]),
                                self,
                                0,
                            )
                        )
                else:
                    if "P" in row[7]:
                        if row[8] != "":
                            games.append(
                                Game(
                                    bool(int(row[1])),
                                    row[2],
                                    None,
                                    bool(int(row[4])),
                                    row[5],
                                    None,
                                    self,
                                    row[8],
                                )
                            )
                        else:
                            games.append(
                                Game(
                                    bool(int(row[1])),
                                    row[2],
                                    None,
                                    bool(int(row[4])),
                                    row[5],
                                    None,
                                    self,
                                    " ",
                                )
                            )
                    else:
                        games.append(
                            Game(
                                bool(int(row[1])),
                                row[2],
                                None,
                                bool(int(row[4])),
                                row[5],
                                None,
                                self,
                                0,
                            )
                        )

        for g in games:
            try:
                g.w()
            except TypeError:
                print(g.t1, g.t2)

    def calculateAccuracy(self):

        with open(self.persistf, "rb") as p:
            (self.teams, self.games) = pickle.load(p)
        rawAccuracy: list[tuple[int, int, float, float]] = []

        begin = self.findFirstFullWeek()

        for g in self.games:
            if g.p_flag:
                currentweek = int(g.week)
            else:
                break

        self.currentweek = currentweek
        games: list[Game] = [g for g in self.games]
        row = 0

        for w in range(begin, currentweek + 1):
            prev_games = Games([g for g in games if g.week < w])
            week_games = Games([g for g in games if g.week == w])

            orig_games = self.games
            orig_teams = self.teams

            teams_temp = Teams(
                [Team(t.codename, t.name, sport=self) for t in orig_teams]
            )
            self.games = prev_games
            temp_teams = Teams()
            for tt in teams_temp:
                temp_teams.append(tt)
            self.teams = temp_teams

            for t in teams_temp:
                t.updatestats()
            for t in teams_temp:
                t.updatemetrics()

            for g in week_games:

                orig_p1, orig_p2 = g.p1, g.p2
                if orig_p1 is None or orig_p2 is None:
                    continue

                g.p1, g.p2 = None, None

                try:
                    g.w()
                except Exception:

                    pass

                try:
                    a = g.w1
                except Exception:
                    a = 0.5

                m = (orig_p1 or 0) - (orig_p2 or 0)
                outcome = int((a > 0.5 and m > 0) or (a < 0.5 and m < 0))
                mae = abs(g.spread2 - m)

                if row:
                    rawAccuracy.append((outcome, int(m > 0), a, mae))
                else:
                    rawAccuracy.append((outcome, int(m < 0), 1 - a, mae))
                row = 1 - row

                g.p1, g.p2 = orig_p1, orig_p2

            self.games = orig_games
            self.teams = orig_teams

            print(
                f"{w}, {avg([i[0] for i in rawAccuracy]):0.4f}, {avg([i[3] for i in rawAccuracy]):0.4f}"
            )

        open(self.accraw, "w").close()
        with open(self.accraw, "a") as b:
            for i in range(len(rawAccuracy)):
                b.write(
                    f"{rawAccuracy[i][0]},{rawAccuracy[i][1]},{rawAccuracy[i][2]:0.2f},{rawAccuracy[i][3]}\n"
                )

        self.rawAccuracy = rawAccuracy
        self.accuracy = avg([i[0] for i in rawAccuracy])
        self.brier = brier(rawAccuracy)

        with open(self.persistf, "wb") as p:
            pickle.dump((self.teams, self.games), p)

    def calculatePlatt(self):
        if not self.rawAccuracy:
            return
        self.log("Scaling...")
        self.platt = platt_scaling(self.rawAccuracy)
        print(f"A: {self.platt[0]:0.4f}, B: {self.platt[1]:0.4f}")
        for g in self.games:
            g.platt = self.platt
            g.w()
        with open(self.persistf, "wb") as p:
            pickle.dump((self.teams, self.games), p)
        self.calculateAccuracy()

    def powm(self):
        m = []

        for t in self.teams:
            m.append(t.power(50))

        for i in m:
            print(*i, sep=",")

    def powm2(self):
        m = []

        for t in self.teams:
            m.append(t.power2())

        for i in m:
            print(*i, sep=",")

    def powerm3(self):
        self.powergames = []

        start = datetime.now()

        for i, teamI in enumerate(self.teams):
            for j in range(i + 1, len(self.teams)):
                teamJ = self.teams[j]
                if not teamJ:
                    continue
                self.powergames.append(
                    Game(
                        False,
                        teamI.codename,
                        None,
                        False,
                        teamJ.codename,
                        None,
                        self,
                        0,
                        datetime.strptime("2000-01-01", "%Y-%m-%d"),
                    )
                )

        for g in self.games:
            for p in self.powergames:
                if (g.t1 == p.t1 and g.t2 == g.p2) or (g.t1 == g.p2 and g.t2 == g.p1):
                    self.powergames.remove(p)

        for g in self.powergames:
            g.w()

        dur = (datetime.now() - start).seconds

        print(dur)

    def playoff(self, tourney=0):

        poff = [[]]
        btemp = []

        if tourney == "nit":
            playoffraw = f"{filepath}\\{self.s}\\{self.year}\\playoff_nit.csv"

        with open(playoffraw, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if row[0] == "0":
                    poff[0].append(None)
                else:
                    poff[0].append(row[0])

        for p in poff[0]:
            if p and p not in self.teams:
                print(f"{p} name error")

        round = int(math.log(len(poff[0]), 2))

        p: list[list[Optional[float]]] = [
            [None for _ in range(round + 1)] for _ in range(len(poff[0]))
        ]

        for n in range(round):
            if len(poff[n]) == 1:
                break
            else:
                poff.append([])
            for i in range(0, len(poff[n]), 2):
                if not poff[n][i + 1]:
                    poff[n + 1].append(poff[n][i])
                else:
                    g = Game(False, poff[n][i], None, False, poff[n][i + 1], None, self)
                    if tourney == "nit" and n < round - 1:
                        g = Game(
                            True, poff[n][i], None, False, poff[n][i + 1], None, self
                        )
                    g.w()
                    try:
                        w = g.w1
                    except AttributeError:
                        g.display()
                    if self.s == "nba":
                        w = self.nbaplayoffgame(poff[n][i], poff[n][i + 1])
                    if w > 0.5:
                        poff[n + 1].append(poff[n][i])
                    else:
                        poff[n + 1].append(poff[n][i + 1])
            self.log(poff[-1])

        for i in range(len(poff[0])):
            p[i][0] = poff[0][i]

        L = list(poff[0])

        for r in range(1, round + 1):
            for i in range(2**round):
                wtemp = []
                n = i // (2**r) * (2**r)
                if i < (n + 2 ** (r - 1)):
                    btemp = list(L[n : n + 2 ** (r + 1)][2 ** (r - 1) : 2**r])
                else:
                    btemp = list(L[n : n + 2 ** (r + 1)][0 : 2 ** (r - 1)])
                for j in btemp:
                    if not L[i]:
                        if len(L) == i + 1:
                            wtemp.append(0.0)
                            break
                        else:
                            if not L[i + 1]:
                                wtemp.append(1.0)
                            else:
                                wtemp.append(0.0)
                    elif L[i] == "x" and not L[i + 1]:
                        if r == 1:
                            wtemp.append(1.0)
                        else:
                            wtemp.append(0.0)
                    else:
                        g = Game(False, L[i], None, False, j, None, self)
                        g.w()
                        v = g.w1
                        if self.s == "nba":
                            v = self.nbaplayoffgame(L[i], j)
                        if r > 1:
                            v = v * (p[poff[0].index(j)][r - 1] or 0)
                        if v > 1:
                            print(L[i], j, v)
                        wtemp.append(v)
                if r > 1:
                    wtemp = sum(wtemp) * (p[i][r - 1] or 0)
                else:
                    wtemp = sum(wtemp)
                p[i][r] = wtemp

        print(round, poff, p)

        with open(playoffraw, "w", newline="") as f:
            csvwriter = csv.writer(
                f, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            for n in range(len(poff[0])):
                if not p[n][0]:
                    csvwriter.writerow([])
                elif p[n][0] == "x":
                    csvwriter.writerow([])
                else:
                    csvwriter.writerow(p[n])

        return poff, p

    def nbaplayoffgame(self, t1: str, t2: str) -> float:
        team1 = self.teams[t1]
        team2 = self.teams[t2]

        if not team1 or not team2:
            return 0.5

        seed1 = float(team1.wl(0))
        try:
            seed2 = float(team2.wl(0))
        except:
            return 1

        g = Game(False, t1, None, False, t2, None, self)
        g.w()

        if seed1 > seed2:
            p = g.w1
        else:
            p = g.w2

        # p = max(p, 1 - p)

        a = min(p + self.ha, 1)
        b = max(p - self.ha, 0)
        c = 1 - a
        d = 1 - b

        x = (
            a**4 * d**3
            + 9 * a**3 * b * c * d**2
            + a**3 * b * d**2
            + 2 * a**3 * b * d
            + 9 * a**2 * b**2 * c**2 * d
            + 6 * a**2 * b**2 * c * d
            + 2 * a**2 * b**2 * c
            + a**2 * b**2
            + a * b**3 * c**3
            + 3 * a * b**3 * c**2
        )

        if seed1 < seed2:
            x = 1 - x

        return x

    def pweight(self, t1: str, s1: int, t2: str, s2: int, ps: str) -> float:
        x = []
        y = []
        p = 0
        q = 0
        g = 0
        for i in self.games:
            if not i.ps:
                for j, v in self.ps.items():
                    if j.upper() in str(i.ps):
                        x.append(i)
                    continue
                continue
        for i in self.games:
            if not i.ps:
                y.append(i)
            continue

        if x:
            p = self.games.index(x[0])
        if y:
            q = self.games.index(y[0])
        p = min(p, q)
        t = len(self.games) - 1
        s = self.ps[ps]

        for i in self.games[p:]:
            if i.t1 == t1 and i.p1 == s1 and i.t2 == t2 and i.p2 == s2:
                g = self.games.index(i)
                for j, v in self.ps.items():
                    if j.upper() in str(i.ps):
                        ps = j
                    continue
                continue
            continue

        return s / 2 + ((s - s / 2) / (math.exp(1) * s**s)) * math.exp(
            2 * s * (g - p) / (t - p)
        )

    def parseGame(self, g, i):
        try:
            self.writeGame(
                ",".join(
                    [
                        str(j)
                        for j in [
                            i,
                            datetime.strptime(g[0], "%Y-%m-%d").strftime("%a %b %d"),
                            "H" if bool(int(g[1])) else "A",
                            g[2],
                            int(g[3]) if g[3] != "" else None,
                            "H" if bool(int(g[4])) else "A",
                            g[5],
                            int(g[6]) if g[6] != "" else None,
                            g[8] if g[8] != "" else " " if "P" in g[7] else 0,
                        ]
                    ]
                )
            )
        except ValueError:
            self.log(f"Game {i} has an unknown problem")

        g = Game(
            bool(int(g[1])),
            g[2],
            int(g[3]) if g[3] != "" else None,
            bool(int(g[4])),
            g[5],
            int(g[6]) if g[6] != "" else None,
            self,
            g[8] if g[8] != "" else " " if "P" in g[7] else 0,
            datetime.strptime(g[0], "%Y-%m-%d"),
            id=i,
        )

        return g

    def writeGame(self, g):
        with open(self.gamesfile, "a") as f:
            f.write(g + "\n")

    def findFirstFullWeek(self):
        tT = [(i.codename, 0) for i in self.teams]
        tT2 = [i.codename for i in self.teams]

        t = 0
        last = 0
        for g in self.games:
            gi = self.games.index(g)
            if self.teams[g.t1] and self.teams[g.t2]:
                t1 = self.teams.index(g.t1)
                t2 = self.teams.index(g.t2)
                if not t1 or not t2:
                    continue
                tT[t1] = (tT[t1][0], tT[t1][1] + 1)
                tT[t2] = (tT[t2][0], tT[t2][1] + 1)
                if tT[t1][1] == 3:
                    tT2.pop(tT2.index(tT[t1][0]))
                    if gi > last:
                        last = self.games.index(g)
                if tT[t2][1] == 3:
                    tT2.pop(tT2.index(tT[t2][0]))
                    if gi > last:
                        last = self.games.index(g)
        self.firstFullWeek = self.games[last].week
        return self.firstFullWeek

    def getDataURL(self):
        return f"https://masseyratings.com/scores.php?s={self.url['url']}{self.year}&sub={self.url['sub']}&all=1&mode=3&sch=on&format=0"

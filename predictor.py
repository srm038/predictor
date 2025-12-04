from datetime import datetime
from dateutil.relativedelta import relativedelta, MO, TH, WE
import math
import pickle
import csv

from sport import Sport
from team import Team
from utils import processRawData


config = {
    "fbs": {
        "url": "cf",
        "sub": "11604",
        "mov": 28,
        "sfile": "NCAA FBS",
        "ps": {"fbs": 5},
        "ha": 0.031,
        "pyth": 2.217,
    },
    "fcs": {
        "url": "cf",
        "sub": "11605",
        "mov": 28,
        "sfile": "NCAA FCS",
        "ps": {"fcs": 3},
        "ha": 0.025,
        "pyth": 1.913,
    },
    "nfl": {
        "url": "nfl",
        "sub": "279539",
        "mov": 21,
        "sfile": "NFL",
        "ps": {"nfl": 3},
        "ha": 0.0445,
        "pyth": 2.583,
    },
    "nba": {
        "url": "nba",
        "sub": "292150",
        "mov": 18,
        "sfile": "NBA",
        "ps": {"nba": 3},
        "ha": 0.0669,
        "pyth": 13.263,
    },
    "mbb": {
        "url": "cb",
        "sub": "11590",
        "mov": 18,
        "sfile": "MBB",
        "ps": {
            "mbb": 1,
            "ncaa": 5,
            "nit": 3,
            "cbi": 1.25,
            "cit": 1.25,
            "vegas": 1.5,
        },
        "ha": 0.0558,
        "pyth": 8.119,
    },
    "wbb": {
        "url": "cbw",
        "sub": "11590",
        "mov": 18,
        "sfile": "WBB",
        "ps": {"wbb": 1, "ncaa": 5, "nit": 3, "wnit": 3, "wbi": 1.25},
        "ha": 0.0505,
        "pyth": 6.557,
    },
    "d1b": {
        "url": "cbase",
        "sub": "11590",
        "mov": 10,
        "sfile": "D1B",
        "ps": {"d1b": 2},
        "ha": 0,
        "pyth": 2,
    },
    "fb": {
        "url": "cfb",
        "sub": "11590",
        "mov": 30,
        "sfile": "FB",
        "ps": {"fb": 1},
        "ha": 0.027,
        "pyth": 2,
    },
    "iru": {
        "url": "iru",
        "sub": "11590",
        "mov": 30,
        "sfile": "IRU",
        "ps": {"rwc": 1},
        "ha": 0.03,
        "pyth": 2,
    },
}


def loadSport():
    sportCode = input("Sport:   ")
    year = input("Year:    ")

    if sportCode in config.keys():
        sport = Sport(
            sportCode,
            year,
            config[sportCode]["mov"],
            config[sportCode]["ps"],
            config[sportCode]["ha"],
            config[sportCode]["pyth"],
        )

    sport.log("\n---\n")
    sport.log(datetime.now().strftime("%m-%d-%y %H:%M:%S"))

    sport.log("Loading data and recalculating")

    processRawData(sport.dataraw)

    with open(sport.teamsraw, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            sport.teams.append(Team(row[0], row[1], sport, row[2]))

    for t in sport.teams:
        t.skins = 1

    with open(sport.dataraw, "r") as f:
        reader = csv.reader(f, delimiter=",")
        i = 1
        for row in reader:
            sport.games.append(sport.parseGame(row, i))
            i += 1
        del i

    sport.log("Redistributing skins")
    for g in sport.games:
        try:
            if g.p_flag and not g.ps:
                if g.p1 > g.p2:
                    sport.teams[g.t1].skins += sport.teams[g.t2].skins
                    sport.teams[g.t2].skins = 0
                elif g.p2 > g.p1:
                    sport.teams[g.t2].skins += sport.teams[g.t1].skins
                    sport.teams[g.t1].skins = 0
        except:
            pass

    open(sport.skinsfile, "w").close()
    for t in sport.teams:
        with open(sport.skinsfile, "a") as f:
            f.write(t.codename + "," + str(t.skins) + "\n")

    for t in sport.teams:
        t.updatestats()
    for t in sport.teams:
        if sport.s == "iru":
            if t.n < 50:
                sport.teams.remove(t)
                sport.NR.append((t.name, t.n))
        elif t.n < 3:
            sport.teams.remove(t)
            sport.NR.append((t.name, t.n))
    for t in sport.teams:
        t.updatemetrics()
    else:
        try:
            maxoeff = max([t.oeff for t in sport.teams])
            minoeff = min([t.oeff for t in sport.teams])
            maxdeff = max([t.deff for t in sport.teams])
            mindeff = min([t.deff for t in sport.teams])
            for t in sport.teams:
                t.oeff = 100 * (t.oeff - minoeff) / (maxoeff - minoeff)
                t.deff = 100 * (t.deff - mindeff) / (maxdeff - mindeff)
        except AttributeError:
            pass

    if sport.s == "fbs":
        sport.dayzero = sport.games[0].day + relativedelta(weekday=MO(-1))
    elif sport.s == "fcs" or sport.s == "iru":
        sport.dayzero = sport.games[0].day + relativedelta(weekday=MO(-1))
    elif sport.s == "nfl":
        sport.dayzero = sport.games[0].day + relativedelta(weekday=TH(-1))
    elif sport.s == "nba":
        sport.dayzero = sport.games[0].day + relativedelta(weekday=WE(-1))
    elif sport.s == "mbb":
        sport.dayzero = sport.games[0].day + relativedelta(weekday=MO(-1))
    elif sport.s == "wbb":
        sport.dayzero = sport.games[0].day + relativedelta(weekday=MO(-1))
    elif sport.s == "d1b":
        sport.dayzero = sport.games[0].day + relativedelta(weekday=MO(-1))
    for g in sport.games:
        g.week = math.floor((g.day - sport.dayzero).days / 7) + 1
        if g.p_flag:
            sport.currentweek = g.week

    try:
        sport.log("Week " + str(sport.currentweek))
    except AttributeError:
        sport.log("Preseason")

    with open(sport.persistf, "wb") as f:
        pickle.dump((sport.teams, sport.games), f)

    sport.maxNameLen = 0
    for i in range(1, len(sport.teams[1:])):
        if len(sport.teams[i].name) > sport.maxNameLen:
            sport.maxNameLen = len(sport.teams[i].name)

    sport.rankteams()
    for n in sport.NR:
        sport.log("Not ranked: ", n)

    return sport


sport = loadSport()

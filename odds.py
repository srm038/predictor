import os
from dotenv import load_dotenv
from math import prod, sqrt
from game import Game
from predictor import sport, loadSport

load_dotenv()

kelly_fraction = float(os.getenv("f") or 0.25)
bankroll = float(os.getenv("BANKROLL") or 0)


def ip(o1, o2):
    return (
        -o1 / (100 - o1) if o1 < 0 else 100 / (o1 + 100),
        -o2 / (100 - o2) if o2 < 0 else 100 / (o2 + 100),
    )


def EF(game: Game, ip: tuple[float, float]) -> tuple[float, float]:
    return game.w1 / ip[0], game.w2 / ip[1]


def OR(o1, o2):
    return (
        1 + (-100 / o1 if o1 < 0 else o1 / 100),
        1 + (-100 / o2 if o2 < 0 else o2 / 100),
    )


def EV(w1, w2, or1, or2):
    return (w1 * (or1 - 1) - w2, w2 * (or2 - 1) - w1)


def f_star(ev1, ev2, or1, or2, kelly_fraction):
    return (ev1 / (or1 - 1) * kelly_fraction, ev2 / (or2 - 1) * kelly_fraction)


def loadOdds() -> list[tuple[str, str, int, int]]:

    with open("odds.tsv", "r") as file:
        odds = [
            tuple(line.strip().split("\t"))
            for line in file.readlines()
            if line.strip() and not line.startswith("#")
        ]
        odds = [(t1, t2, int(o1), int(o2)) for t1, t2, o1, o2 in odds]

    return odds


def runAnalysis(odds: list[tuple[str, str, int, int]]) -> list[dict]:

    analyses = []

    for g in odds:
        t1, t2, o1, o2 = g
        game = Game(t1=t1, t2=t2, h1=False, h2=True, p1=None, p2=None, sport=sport)
        game.w()

        analysis: dict = {"game": game}

        analysis["odds"] = (o1, o2)
        analysis["ip"] = ip(o1, o2)
        analysis["EF"] = EF(game, analysis["ip"])
        analysis["OR"] = OR(o1, o2)
        analysis["EV"] = EV(game.w1, game.w2, analysis["OR"][0], analysis["OR"][1])
        analysis["f*"] = f_star(
            analysis["EV"][0],
            analysis["EV"][1],
            analysis["OR"][0],
            analysis["OR"][1],
            kelly_fraction,
        )

        analyses.append(analysis)

    for a in sorted(analyses, key=lambda x: max(x["f*"]), reverse=True):
        if a["EV"][0] <= 0.2 and a["EV"][1] <= 0.2:
            continue
        warning = (a["odds"][0] > 0 and a["game"].w1 > 0.5) or (
            a["odds"][1] > 0 and a["game"].w2 > 0.5
        )
        if a["EV"][0] > 0:
            if a["f*"][0] * bankroll >= 0.1:
                print(
                    f"Bet ${a['f*'][0] * bankroll:0.2f} ({a['EV'][0]:0.2f}, {a['EF'][0]:0.2f}) on {a['game'].t1} (@ {a['game'].t2}) {'\uEA6C' if warning else ''}"
                )
        if a["EV"][1] > 0:
            if a["f*"][1] * bankroll >= 0.1:
                print(
                    f"Bet ${a['f*'][1] * bankroll:0.2f} ({a['EV'][1]:0.2f}, {a['EF'][1]:0.2f}) on {a['game'].t2} (v {a['game'].t1}) {'\uEA6C' if warning else ''}"
                )

    return analyses


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
        tef = []
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
            tef.append(g["EF"][pick[-1]])
            tev.append(g["EV"][pick[-1]])
            tor.append(g["OR"][pick[-1]])
        if not skip:
            p.update({"tp": prod(tp)})
            p.update({"tip": prod(tip)})
            p.update({"tev": sum(tev)})
            p.update({"tef": prod(tef)})
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
                f"Bet ${p['f*'] * bankroll:0.2f} ({p['tev']:0.2f}, {p['tef']:0.2f}) {' + '.join(picks)}"
            )

    return parlays

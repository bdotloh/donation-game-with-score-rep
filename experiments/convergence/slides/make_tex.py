"""
Build the slides from the numbers money.py saved.

Reads money_results.json, writes results.tex (the \\res... macros that money_slides.tex inputs) and
compiles money_slides.tex if latexmk is installed. The figures are drawn by money.py.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

with open(HERE / "money_results.json") as fh:
    res = json.load(fh)
p, sim1, sim2, exact1, exact2, w0 = (res[k] for k in ("params", "sim1", "sim2", "exact1", "exact2", "w0"))


def num(x, digits=4):
    return "--" if x is None or np.isnan(x) else f"{x:.{digits}f}"


def matrix(M, names):
    rows = [" & ".join([names[i]] + [f"${num(x)}$" for x in M[i]]) + r" \\" for i in range(len(names))]
    return "\n".join([r"\begin{tabular}{l" + "r" * len(names) + "}", r"\toprule",
                      " & ".join([""] + names) + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def steps(t):
    return f"{int(t):,}" if t >= 0 else "not reached"


names = ["disc", "ternary"]
macros = {
    "tau": p["tau"], "eps": p["eps"], "epsup": p["eps_up"], "epsdown": num(p["eps_down"], 2), "rho": p["rho"],
    "benefit": f"{p['benefit']:g}", "cost": f"{p['cost']:g}", "tol": f"10^{{{int(np.log10(p['tol']))}}}",
    "stepsOne": steps(sim1["t_conv"][0]),
    "lamOne": num(sim1["lam"][0], 6), "lamLinOne": num(sim1["lam_lin"][0], 6),
    "trelTwo": f"{-1 / np.log(sim2['lam'][0]):,.0f}".replace(",", "{,}"),
    "HOne": num(sim1["H"][0][0]), "POne": num(sim1["P"][0][0]), "fitOne": num(sim1["fitness"][0]),
    "stepsTwoA": steps(sim2["t_conv"][0]), "stepsTwoB": steps(sim2["t_conv"][1]),
    "targetOne": steps(sim1["t_target"][0]),
    "targetTwoA": steps(sim2["t_target"][0]), "targetTwoB": steps(sim2["t_target"][1]),
    "lamTargetTwoA": num(sim2["lam_target"][0], 6), "lamTargetTwoB": num(sim2["lam_target"][1], 6),
    "stepsTwo": f"{sim2['n_iter']:,}",
    "lamTwoA": num(sim2["lam"][0], 6), "lamTwoB": num(sim2["lam"][1], 6),
    "lamLinTwoA": num(sim2["lam_lin"][0], 6), "lamLinTwoB": num(sim2["lam_lin"][1], 6),
    "fitTwoA": num(sim2["fitness"][0]), "fitTwoB": num(sim2["fitness"][1]),
    "helpTwo": num(sim2["help_rate"]),
    "wzeroOne": num(w0["pi_1"]), "wzeroTwoA": num(w0["pi_2A"]), "wzeroTwoB": num(w0["pi_2B"]),
    "rZero": num(p["r0"]), "rOne": num(exact1["r"]), "rTwo": num(exact2["r"]), "sTwo": num(exact2["s"]),
    "gOne": num(exact1["gA"]), "fOne": num(exact1["fA"]), "gTwo": num(exact2["gA"]), "fTwo": num(exact2["fA"]),
    "gTauTwo": num(exact2["gB_tau"]),
    "wzeroExactOne": num(exact1["w0A"]), "wzeroExactTwoA": num(exact2["w0A"]), "wzeroExactTwoB": num(exact2["w0B"]),
}
with open(HERE / "results.tex", "w") as fh:
    fh.write("% written by make_tex.py from money_results.json\n")
    for key, value in macros.items():
        fh.write(f"\\newcommand{{\\res{key[0].upper() + key[1:]}}}{{{value}}}\n")
    fh.write("\\newcommand{\\resHTwo}{%\n" + matrix(sim2["H"], names) + "}\n")
    fh.write("\\newcommand{\\resPTwo}{%\n" + matrix(sim2["P"], names) + "}\n")

if shutil.which("latexmk"):
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "money_slides.tex"],
                   cwd=HERE, check=True)
    print("slides:", HERE / "money_slides.pdf")
else:
    print("latexmk not found; compile", HERE / "money_slides.tex", "by hand")

"""Writes the checks of the verification report (checks.tex) and its summary table (summary.tex).

    python docs/verification/build_report.py && cd docs/verification && pdflatex report.tex

Every check has the same fields, so the report reads the same way throughout:

    kind      the kind of check (one row of the summary table)
    title     what is being checked
    what      what was checked
    against   what it was checked against
    count     how many things were checked
    result    "Agree" (no difference above what the check allows: rounding error for exact calculations, sampling
              noise for the simulation), "Agree*" (with an explained exception) or "Open" (they differ and it is
              not explained)
    note      anything else worth knowing (optional)
    figure    (file, caption) (optional)

The numbers are those of the runs described in the notes of each check. The test inventory is counted from pytest.
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

JULIA = r"the output of \texttt{helper\_funcs.jl} (saved by \texttt{make\_reference.jl})"
INDEP = (r"an independent re-implementation of the model written from the notes and \texttt{helper\_funcs.jl} "
         r"without looking at the engine's code (\texttt{tests/independent\_model.py})")

CHECKS = [
    # ---------------------------------------------------------------- the Julia code
    dict(kind="Binary engine", title="Stable scores, payoffs and rates against the Julia code",
         what=r"The engine's final score distributions, payoffs and the three rates $\alpha,\beta,\gamma$ of binary "
              r"strategies and norms.",
         against=JULIA + ".",
         count=r"400 cases: 200 with one strategy and 200 with two, at noise 0, 0.01 and 0.1, run for 0 to 1000 steps.",
         result="Agree",
         note=r"Payoffs follow Julia's convention: one strategy from the distribution before the last step, two "
              r"strategies from the final one."),
    dict(kind="Binary ESS", title="The ESS search against the Julia code",
         what=r"The list of ESS that \texttt{find\_ESS} finds among the 16 binary strategies for a norm "
              r"($\rho_A=0.99$, 100 steps, fitness threshold $10^{-5}$).",
         against=JULIA + ".",
         count=r"3192 rows (1596 norms at noise 0 and at noise 0.01), 809 ESS in all.",
         result="Agree", note=r"Every row is compared by \texttt{tests/reference/compare\_ess.py}; the test suite checks a "
                              r"sample of 60."),
    dict(kind="ESS sweep script", title="The sweep script against the Julia code",
         what=r"The ESS found by the sweep script (the one that runs on R\=apoi).",
         against=r"The Julia ESS lists for the same norms.",
         count=r"1296 CN-only binary norms at noise 0 and at noise 0.01.", result="Agree"),
    dict(kind="Lifted ternary", title="A ternary norm that ignores the cut-off against the binary result",
         what=r"A binary norm and binary strategies lifted to ternary classes must give exactly the binary results.",
         against=r"The binary engine.",
         count=r"12 test cases (4 norms $\times$ $\tau\in\{1,2,5\}$), and an ESS sweep of the 1296 CN-only norms lifted "
               r"to ternary.", result="Agree"),
    # ---------------------------------------------------------------- the notes
    dict(kind="Notes: Defectors and Discriminators", title="The worked example of pages 5 and 6",
         what=r"The stationary distributions of Defectors and Discriminators at $\epsilon=0.1$, $\epsilon_\uparrow=0.4$, "
              r"$\epsilon_\downarrow=0.6$, and the comparison of 2000 steps with them.",
         against=r"The closed-form numbers printed in the notes: $\pi_0=1/3$ and $0.754095$, $q(\boldsymbol\pi_0)=0.456286$, "
                 r"mean scores $0.326093$ and $2$, and the differences $1.10\times10^{-5}$ and $7.36\times10^{-5}$.",
         count=r"7 quantities from the notes, and both distributions at scores 0 to 20.", result="Agree",
         note=r"The notes do not say how the 2000 steps start; the initial distribution $(0.5,0.5)$ reproduces their "
              r"differences.",
         figure=("figures/defectors_discriminators.png",
                 r"Defectors and Discriminators. Left: closed-form stationary probabilities (lines) and the engine "
                 r"after 2000 steps (circles). Right: the absolute differences.")),
    dict(kind="Notes: $\\phi$-Disc rates", title=r"The $\phi$-Disc rates of page 7",
         what=r"The up- and down-rates $g_j$ and $f_j$ of a population of $\phi$-Disc players under "
              r"\texttt{CN:00000-0+ NC:00000000}.",
         against=r"The formulas of the notes, $G_j=\frac{1-w_0}{4(j-1)^{\phi}}$ and "
                 r"$F_j=\frac{1-w_0}{4}(1-j^{-\phi})$, with Eqs.~3--4.",
         count=r"5 values of $\phi$ (0, 0.17, 0.5, 1, 2) at scores 1 to 7, and 1000 random settings ($\phi\in[0,3]$, "
               r"noise, score distributions).", result="Agree",
         note=r"The notes print the norm as \texttt{CN:000000-+}, which is not a valid norm (a score-0 donor would "
              r"lose a score); \texttt{CN:00000-0+} is meant and is what reproduces the notes' numbers."),
    dict(kind="Notes: $\\phi$-Disc, $\\phi=1$", title=r"The $\phi=1$ stationary distribution of page 8",
         what=r"The stationary score distribution of $\phi$-Disc players at $\epsilon=0.1$, $\epsilon_\uparrow=0.4$, "
              r"$\epsilon_\downarrow=0.6$, and its $\pi_0$ and mean score.",
         against=r"The gamma-function solution of the notes ($\pi_0\approx0.2555$, $\mathbb{E}[\text{score}]=1.889$), "
                 r"with $\pi_0$ from the normalisation equation.",
         count=r"1 distribution at all 440 tracked scores; 6 further settings of the noise.", result="Agree",
         figure=("figures/phi_disc.png",
                 r"$\phi$-Disc with $\phi=1$. Left: the gamma-function solution (line) and the engine after 2000 steps "
                 r"from score 0 (circles). Right: the absolute differences.")),
    dict(kind="Notes: $\\phi=0$", title=r"$\phi=0$ is the Discriminators",
         what=r"The rates of $\phi$-Disc with $\phi=0$ and of Discriminators in the same population.",
         against=r"Each other (the notes say the two coincide).",
         count=r"500 random settings of the norm, noise and score distribution.", result="Agree"),
    dict(kind="Notes: stationary equations", title="The fixed-point equation for $\\pi_0$ (Eq.~11)",
         what=r"$\pi_0=\dfrac{f-g}{f-g+g_1}$ evaluated at the engine's converged distribution.",
         against=r"The engine's own $\pi_0$.",
         count=r"108 chains from 150 random binary settings with one or two strategies (the 73 settings that never "
               r"settled within 4000 steps were left out).", result="Agree",
         note=r"The detailed-balance equations (Eqs.~5--6) are checked on the stationary distributions in Check~12."),
    # ---------------------------------------------------------------- independent models
    dict(kind="Independent model: rates", title=r"The rates $G,F,g,f$ against an independent implementation",
         what=r"The transition rates for random populations of binary, ternary and $\phi$-Disc strategies.",
         against=INDEP + ".",
         count=r"5000 random configurations: binary, ternary and lifted norms; $\tau\in\{1,2,3,5,8\}$; one to three "
               r"strategies of mixed kinds (probabilistic and deterministic tables, $\phi\in[0,3]$); noise from 0 to 1; "
               r"shares of zero; score distributions with mass beyond $\tau$.", result="Agree"),
    dict(kind="Independent model: chains", title="The score chains over time against the independent implementation",
         what=r"The score distributions after 25 to 400 steps of the chain, with mass conserved.",
         against=r"The independent implementation iterating the same chain.",
         count=r"1780 chains.", result="Agree"),
    dict(kind="Independent model: stationary", title="Stationary distributions",
         what=r"Whether a stationary distribution found independently (Eqs.~5--6 and a fixed point on the class "
              r"masses) is left unchanged by one engine step, and whether the engine's own converged runs reach it.",
         against=r"The independent implementation.",
         count=r"227 random settings for the first, 107 for the second.", result="Agree",
         note=r"How close a converged run gets is set by the engine's stopping rule, not by the model."),
    dict(kind="Closed form: ternary", title="Three geometric segments for a class-based ternary strategy",
         what=r"The stationary distribution of one ternary strategy, which is geometric over the three classes.",
         against=r"The closed form, with the class masses solved self-consistently, independently of the engine.",
         count=r"100 configurations ($\tau\in\{1,2,3,5,8\}$).", result="Agree"),
    dict(kind="Mismatched classes", title="Strategies whose classes differ from the norm's",
         what=r"Rates when a strategy's cut-off $\tau$ differs from the norm's, or a binary strategy meets a ternary "
              r"norm and the other way round (the engine uses the raw scores).",
         against=r"The independent implementation.", count=r"2100 configurations.", result="Agree"),
    dict(kind="Edge cases", title="Awkward settings",
         what=r"Rates and chains at $\tau=1$, with an empty top class, with $\epsilon=0$, and with a strategy whose "
              r"share is 0.",
         against=r"The independent implementation (a population with and without the strategy of share 0).",
         count=r"$\tau=1$: 300; empty top class: 200; $\epsilon=0$: 159; share 0: 200.", result="Agree"),
    dict(kind="Drifting flag", title="Which chains are flagged as drifting",
         what=r"That the engine flags a chain exactly when its up-rate at the top score is positive and at least its "
              r"down-rate.",
         against=r"The rule, evaluated independently.", count=r"833 configurations.", result="Agree*",
         note=r"When nothing can move (both rates 0) the engine does not flag the chain; the README now says so."),
    dict(kind="Agent-based simulation", title="A simulated population against the engine",
         what=r"Score histograms after the same number of steps: a population of $10^6$ agents meeting random "
              r"partners, with roles, help, the norm and noise drawn as in the model, against the engine.",
         against=r"The engine, bin by bin, with binomial standard errors.",
         count=r"470 comparisons (the two example runs, $\phi$-Disc alone, Defectors against Discriminators, random "
               r"ternary norms, a mix of $\phi$-Disc and a ternary strategy); 11 stationary scenarios, 48 comparisons.",
         result="Agree*",
         note=r"Two comparisons, both in one random case that the engine flags as drifting, fall outside the noise: "
              r"the default stopping tolerance leaves the engine short of the stationary law there. The same test "
              r"detects a 10\% error in $\epsilon$ and a 4\% error in the role factor."),
    dict(kind="Hand-computed cases", title="Small cases worked out on paper",
         what=r"Rates for tiny populations at $\tau=1$ and $2$: CN against NC, the recipient's index against the "
              r"donor's, the class boundary at scores $\tau$ and $\tau+1$, the noise arithmetic, the $\phi$-Disc "
              r"exponent, one step of the chain.",
         against=r"Numbers written as literal arithmetic from the notes.", count=r"18 cases.", result="Agree"),
    dict(kind="Planted bugs", title="Would the checks notice a mistake?",
         what=r"Whether the test suites fail when a bug is planted in a copy of the engine (a wrong cut-off, a "
              r"swapped CN and NC, a transposed table, a wrong lift to ternary, the wrong score in the $\phi$ "
              r"exponent, a wrong role factor, \dots).",
         against=r"Four suites: the hand cases, the independent tests, the repository's own tests and the simulation "
                 r"tests.",
         count=r"25 planted bugs.", result="Agree",
         note=r"Every bug was caught by at least one suite. No single suite caught them all (the repository's own "
              r"tests missed three cut-off bugs; the simulation tests missed a transposed table and a wrong lift), "
              r"which is why all four are kept."),
    # ---------------------------------------------------------------- the paper's code and data
    dict(kind="Leading Eight", title="The Leading Eight norms",
         what=r"The barcodes, names and ESS strategies of L1 to L8 in \texttt{known\_norms.py}, and whether "
              r"the engine finds a fully cooperative ESS for each.",
         against=r"\texttt{createLeading8()} in the paper's code, and the engine's own ESS search (noise 0.001 and "
                 r"0.01).",
         count=r"8 norms: barcodes and strategies equal; a cooperative ESS found for 8 of 8.", result="Agree",
         note=r"The five entries all eight share and the three that vary are also checked in the tests."),
    dict(kind="ESS figure", title="The ESS figure against the paper's",
         what=r"\texttt{plots/ess\_plot.py}: the layout, sort order, cell widths, colours and Leading Eight "
              r"labelling.",
         against=r"The paper's \texttt{plotting\_funcs.py} and the figures it produced.",
         count=r"2 data files drawn and compared by eye (25 and 403 rows), and @plots@ unit tests.", result="Agree"),
    dict(kind="Published ESS data", title="The paper's published ESS table",
         what=r"The ESS the engine finds for every CN-only norm (1296) at noise 0.01, 500 steps and the paper's "
              r"fitness threshold 0.01, compared by norm and strategy, and their cooperation levels.",
         against=r"The paper's file \texttt{tokens\_CmeetsN\_Noise0.01\_B2} (its header says 500 steps).",
         count=r"403 published ESS rows (302 norms); the same sweep with 200 and 100 steps.", result="Agree*",
         note=r"The engine finds 359 of the 403 and no ESS that the file lacks; the cooperation levels of the 359 "
              r"differ by at most $8\times10^{-4}$. The other 44 (in 25 norms) are payoff ties: the invader's payoff "
              r"is larger than the resident's by at most $3.3\times10^{-16}$, because strategies that differ only "
              r"where it never matters earn the same payoff and rounding decides. Running 200 steps does not come "
              r"closer: it finds 362 but adds 33 ESS that the file lacks and the cooperation levels differ by up to "
              r"0.27 (100 steps: 346 found, 29 extra, up to 0.45). The file's header says 500 steps."),
    # ---------------------------------------------------------------- the scripts
    dict(kind="Script: Markov chain", title="\\texttt{run\\_markov\\_chain.py} on the engine",
         what=r"Rates, class masses and score distributions of every step, and the number of steps.",
         against=r"The old script and the seven run directories committed in the repository.",
         count=r"20 argument sets, 1800 random norms, strategies and initial distributions, and 7 committed runs.",
         result="Agree*",
         note=r"The old script capped scores at \texttt{maxw}; the engine does not. Where the cap did not bind the two "
              r"agree. Where it did, the rates and class masses still agree in the committed runs while the score "
              r"distributions continue above the cap; with a very small cap the old results differ."),
    dict(kind="Script: ESS sweep (old)", title="\\texttt{experiments/ESS/find\\_ESS\\_parellel.py} on the engine",
         what=r"The ESS found for every CN-only norm, and the CSV written.",
         against=r"The old code, the committed CSV and the Julia ESS lists.",
         count=r"1296 norms (333 ESS rows); the ternary path in 64 cases.", result="Agree",
         note=r"The old code took two-strategy payoffs from the distribution before the last step and the engine, "
              r"like Julia, from the final one; this only reorders near-ties and changes no ESS."),
    dict(kind="Script: zeroness", title="\\texttt{experiments/verify-zeroness-one-strategy} on the engine",
         what=r"The group of initial rates, the final rates and the share of score 0 of every chain, and the limit-check "
              r"tables.",
         against=r"The old code and the committed tables.",
         count=r"20736 chains (1296 norms $\times$ 16 strategies); the 20736-row limit-check table; the rate function "
               r"at 11 class masses for every chain.", result="Agree*",
         note=r"The old chains were capped at \texttt{maxw}, so about half of them now track more scores; the limit "
              r"check keeps its own capped chain."),
    # ---------------------------------------------------------------- the rest
    dict(kind="Replicator dynamics and rest points", title="Games with known answers",
         what=r"Rest points and their stability, the replicator equation, and the restart of the score chains.",
         against=r"Games whose answers are known (prisoner's dilemma, hawk--dove, coordination with two and three "
                 r"strategies, rock--paper--scissors in both versions), the invasion fitness of the engine's own "
                 r"payoffs, and the analytic Cooperators against Defectors.",
         count=r"@dynamics@ tests (@simplex@ for the simplex and rest points, @replicator@ for the dynamics).", result="Agree"),
    dict(kind="GUI", title="The GUI logic and a click-through in a browser",
         what=r"Input checks, settings files, presets, figures, the strategies and norm cards, and a click through "
              r"all three pages in Chromium (including going back to the results, typing axis ranges and saving).",
         against=r"The expected behaviour.",
         count=r"@gui@ unit tests, and @e2e@ browser tests (full click-through, two strategies from a settings file, "
               r"save and load).", result="Agree"),
    dict(kind="Saved results", title="Saving a result and loading it again",
         what=r"That a loaded result has every number of the saved one, that a new point can still be computed, "
              r"that the file is plain queryable tables, and that a taken name, a damaged file and a foreign file "
              r"are refused.",
         against=r"The finished run that was saved (2 and 3 strategies).",
         count=r"@saves@ tests; every point on the simplex, every trajectory step, every matrix and distribution.",
         result="Agree",
         note=r"The far tail of each score distribution (probabilities below $10^{-12}$) is not stored and reads "
              r"back as 0."),
    dict(kind="Portability", title="Other versions of Python",
         what=r"That the engine and the sweep run on the Python that R\=apoi provides.",
         against=r"Python 3.9 with numpy 1.26 and scipy 1.11, and Python 3.13 with numpy 2.5 and scipy 1.18.",
         count=r"2 environments: 517 engine tests and the sweep tests on 3.9; the whole suite on 3.13.",
         result="Agree"),
]

TEST_FILES = [
    ("test_against_julia.py", "400 stable-score cases, 60 sampled ESS rows"),
    ("test_engine.py", "norms, strategies, closed forms, invariants"),
    ("test_independent_model.py", "independent re-implementation"),
    ("test_agent_based.py", "agent-based simulation"),
    ("test_hand_cases.py", "18 hand-computed cases"),
    ("test_simplex.py", "grid, rest points, stability"),
    ("test_replicator.py", "dynamics, restart, role factor"),
    ("test_known_norms.py", "Leading Eight"),
    ("test_plots.py", "ESS figure and readers"),
    ("test_saves.py", "saving and loading results"),
    ("test_gui_logic.py", "GUI logic"),
    ("test_ess_sweep.py", "ESS sweep script"),
    ("test_experiments.py", "migrated experiment scripts"),
    ("test_run_markov_chain.py", "Markov chain script"),
]


def counts():
    root = HERE.parents[1]
    found = {}
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "not e2e"], cwd=root,
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        match = re.match(r"tests/(test_\w+\.py)::", line)
        if match:
            found[match.group(1)] = found.get(match.group(1), 0) + 1
    e2e = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "e2e"], cwd=root,
                         capture_output=True, text=True).stdout
    found["e2e"] = len(re.findall(r"^tests/test_gui_e2e\.py::", e2e, flags=re.M))
    return found


COUNTS = counts()
FILL = {"@gui@": COUNTS["test_gui_logic.py"], "@e2e@": COUNTS["e2e"], "@saves@": COUNTS["test_saves.py"],
        "@plots@": COUNTS["test_plots.py"], "@simplex@": COUNTS["test_simplex.py"],
        "@replicator@": COUNTS["test_replicator.py"],
        "@dynamics@": COUNTS["test_simplex.py"] + COUNTS["test_replicator.py"]}


def fill(text):
    for key, value in FILL.items():
        text = text.replace(key, str(value))
    return text


def check_blocks():
    out = []
    for n, c in enumerate(CHECKS, start=1):
        out.append(f"\\subsection*{{Check {n}: {c['title']}}}\n\\addcontentsline{{toc}}{{subsection}}{{Check {n}: {c['title']}}}\n")
        out.append("\\begin{description}\\setlength\\itemsep{0pt}\n")
        out.append(f"\\item[What was checked] {c['what']}\n")
        out.append(f"\\item[Checked against] {c['against']}\n")
        out.append(f"\\item[How many] {c['count']}\n")
        out.append(f"\\item[Result] \\textbf{{{c['result'].replace('*', '')}}}"
                   + (" (with the exception below)" if c["result"] == "Agree*" else "")
                   + (" (not yet explained)" if c["result"] == "Open" else "") + "\n")
        if c.get("note"):
            out.append(f"\\item[Note] {c['note']}\n")
        out.append("\\end{description}\n")
        if c.get("figure"):
            path, caption = c["figure"]
            out.append("\\begin{figure}[h]\n\\centering\n"
                       f"\\includegraphics[width=\\textwidth]{{{path}}}\n\\caption{{{caption}}}\n\\end{{figure}}\n")
        out.append("")
    return fill("\n".join(out))


def summary_table():
    rows = []
    for n, c in enumerate(CHECKS, start=1):
        rows.append(f"{n} & {c['kind']} & {short_against(c)} & {short_count(c)} & {c['result']} \\\\")
    return ("\\begin{longtable}{@{}r>{\\raggedright\\arraybackslash}p{4.2cm}>{\\raggedright\\arraybackslash}p{4.6cm}"
            ">{\\raggedright\\arraybackslash}p{3.3cm}l@{}}\n\\toprule\n"
            "\\# & Kind of check & Checked against & How many & Result \\\\\n\\midrule\n\\endhead\n"
            + fill("\n".join(rows)) + "\n\\bottomrule\n\\end{longtable}\n"
            "Agree = no difference above what the check allows (rounding error for exact calculations, sampling "
            "noise for the simulation). Agree* = agree, with an exception explained in the check. "
            "Open = they differ and it is not explained.\n")


SHORT = {
    1: ("Julia code", "400 cases"), 2: ("Julia code", "3192 rows"), 3: ("Julia code", "1296 norms $\\times$ 2 noise levels"),
    4: ("binary engine", "12 cases and 1296 norms"), 5: ("the notes, pp.~5--6", "7 quantities"),
    6: ("the notes, p.~7", "5 + 1000 settings"), 7: ("the notes, p.~8", "1 distribution, 6 settings"),
    8: ("the notes", "500 settings"), 9: ("the notes, Eq.~11", "108 chains"),
    10: ("independent model", "5000 configurations"), 11: ("independent model", "1780 chains"),
    12: ("independent model", "227 + 107 settings"), 13: ("closed form", "100 configurations"),
    14: ("independent model", "2100 configurations"), 15: ("independent model", "859 settings"),
    16: ("the rule", "833 configurations"), 17: ("simulation, $10^6$ agents", "470 comparisons"),
    18: ("hand arithmetic", "18 cases"), 19: ("4 test suites", "25 planted bugs"),
    20: ("the paper's code", "8 norms"), 21: ("the paper's code", "2 figures, @plots@ tests"),
    22: ("the paper's data", "403 ESS rows"), 23: ("old script, committed runs", "20 + 1800 + 7"),
    24: ("old script, Julia", "1296 norms"), 25: ("old script, committed tables", "20736 chains"),
    26: ("games with known answers", "@dynamics@ tests"), 27: ("expected behaviour", "@gui@ unit tests, @e2e@ browser tests"),
    28: ("the saved run", "@saves@ tests"), 29: ("Python 3.9 and 3.13", "2 environments"),
}


def short_against(c):
    return SHORT[CHECKS.index(c) + 1][0]


def short_count(c):
    return SHORT[CHECKS.index(c) + 1][1]


def inventory():
    rows = [f"\\texttt{{{name.replace('_', chr(92) + '_')}}} & {COUNTS.get(name, 0)} & {what} \\\\"
            for name, what in TEST_FILES]
    total = sum(COUNTS.get(name, 0) for name, _ in TEST_FILES)
    return (total, COUNTS["e2e"], "\\begin{tabular}{@{}lrl@{}}\n\\toprule\nFile & Tests & Covers \\\\\n\\midrule\n"
            + "\n".join(rows) + f"\n\\midrule\nTotal & {total} & \\\\\n\\bottomrule\n\\end{{tabular}}\n")


if __name__ == "__main__":
    assert len(SHORT) == len(CHECKS), "the summary table needs a short form for every check"
    (HERE / "checks.tex").write_text(check_blocks())
    (HERE / "summary.tex").write_text(summary_table())
    total, e2e, table = inventory()
    (HERE / "inventory.tex").write_text(table)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    results = [c["result"] for c in CHECKS]
    (HERE / "counts.tex").write_text(
        f"\\newcommand{{\\ntests}}{{{total}}}\n\\newcommand{{\\nchecks}}{{{len(CHECKS)}}}\n"
        f"\\newcommand{{\\nbrowser}}{{{e2e}}}\n\\newcommand{{\\nagree}}{{{results.count('Agree')}}}\n"
        f"\\newcommand{{\\nexception}}{{{results.count('Agree*')}}}\n\\newcommand{{\\nopen}}{{{results.count('Open')}}}\n"
        f"\\newcommand{{\\commit}}{{{commit}}}\n")
    print(f"{len(CHECKS)} checks, {total} tests, {e2e} browser tests")

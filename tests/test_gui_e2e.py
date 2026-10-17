"""Click through the GUI in a real browser (Chromium via Playwright).

Run with:  python -m pytest -m e2e tests/test_gui_e2e.py
Screenshots go to $GUI_SCREENSHOTS (default: a temporary directory).
"""

import glob
import os
import re
import tempfile
import threading
import time

import pytest

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e

PORT = 8765


def _chromium():
    for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux*/chrome",):
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    return None


@pytest.fixture(scope="module")
def server():
    from werkzeug.serving import make_server

    import logging

    from gui.app import create_app

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    app = create_app()
    srv = make_server("127.0.0.1", PORT, app.server, threaded=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{PORT}"
    srv.shutdown()


@pytest.fixture(scope="module")
def page(server):
    shots = os.environ.get("GUI_SCREENSHOTS") or tempfile.mkdtemp()
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_chromium())
        context = browser.new_context(viewport={"width": 1400, "height": 1000}, accept_downloads=True)
        pg = context.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.shots = shots
        pg.errors = errors
        pg.goto(server)
        yield pg
        browser.close()


def dump_threads():
    """Print the stacks of all threads (to diagnose a hung computation)."""
    import sys
    import traceback

    for tid, frame in sys._current_frames().items():
        print(f"--- thread {tid}\n" + "".join(traceback.format_stack(frame)[-8:]))


def shot(page, name):
    page.screenshot(path=os.path.join(page.shots, f"{name}.png"), full_page=True)


def choose(page, dropdown_id, option_text):
    page.click(f"#{dropdown_id}")
    page.get_by_role("option", name=option_text, exact=True).click()
    page.wait_for_load_state("networkidle")          # let the callbacks that follow (the preview) finish


def click_plot(page, selector, fx, fy):
    """Hover, then click, at a fraction (fx, fy) of a plot area (Plotly clicks the hovered point)."""
    page.locator(selector).scroll_into_view_if_needed()          # the cards above the plot push it down
    box = page.locator(selector).bounding_box()
    x, y = box["x"] + fx * box["width"], box["y"] + fy * box["height"]
    page.mouse.move(x - 5, y)
    page.mouse.move(x, y)
    time.sleep(0.3)
    page.mouse.click(x, y)


def set_text(page, selector, value):
    box = page.locator(selector)
    box.fill("")
    box.fill(str(value))


def test_full_flow(page):
    try:
        _full_flow(page)
    except Exception:
        shot(page, "failure")
        dump_threads()
        raise


def _full_flow(page):
    # Page 1: strategies.
    page.wait_for_selector("#s-preview .main-svg")
    choose(page, "s-preset", "AllD-Disc")
    page.wait_for_function("document.querySelector('#s-name').value === 'AllD-Disc'")
    page.wait_for_load_state("networkidle")
    page.click("#s-add")
    choose(page, "s-preset", "Defectors")
    page.wait_for_function("document.querySelector('#s-name').value === 'Defectors'")
    page.click("#s-add")
    choose(page, "s-type", "Ternary (score 0 / 1 to τ / above τ)")
    page.wait_for_selector("#s-tau-row:not([style*='none'])")
    page.wait_for_selector(".cell input >> nth=8")
    assert page.locator(".cell input").count() == 9
    wanted = ["0", "0", "0", "0", "1", "0.5", "0", "1", "1"]
    for _ in range(4):          # a callback that redraws the grid can overwrite a value typed just before: retype
        for k, value in enumerate(wanted):
            if page.input_value(f".cell input >> nth={k}") != value:
                set_text(page, f".cell input >> nth={k}", value)
        page.wait_for_load_state("networkidle")
        if [page.input_value(f".cell input >> nth={k}") for k in range(9)] == wanted:
            break
    set_text(page, "#s-name", "Ternary")
    page.wait_for_load_state("networkidle")            # the preview has to catch up with what was typed
    page.wait_for_function("!document.querySelector('#s-add').disabled")
    for _ in range(9):                      # a click on Add is now and then lost, or slow to show: click again
        if page.locator(".selected-item").count() >= 3:
            break
        if not page.locator("#s-add").is_disabled():
            page.click("#s-add")
        try:
            page.wait_for_selector(".selected-item >> nth=2", timeout=10_000)
        except playwright.TimeoutError:
            pass
    page.wait_for_selector(".selected-item >> nth=2", timeout=10_000)
    assert "0,0,0|0,g,0.5|0,g,g (tau=2)" in page.inner_text(".selected-item >> nth=2")   # stored as typed
    assert page.locator("#s-add").is_disabled()           # at most 3
    shot(page, "1_strategies")
    page.click("#s-next")

    # Page 2: parameters.
    page.wait_for_selector("#p-norm-grid table")
    assert page.locator("input[id*='p-share']").count() == 0          # the starting point is chosen on the results page
    assert page.locator("#p-steps").count() == 0 and page.locator("#p-dt").count() == 0
    set_text(page, "#p-score", "2")
    set_text(page, "#p-mass", "0.2")
    page.click("#p-add-score")
    page.wait_for_selector("#p-score-table .status.error")  # 0.5 + 0.5 + 0.2 != 1
    page.locator("#p-score-table button").first.click()      # remove score 0
    page.wait_for_function("document.querySelectorAll('#p-score-table tr').length === 3")
    set_text(page, "#p-score", "0")
    set_text(page, "#p-mass", "0.3")
    page.click("#p-add-score")
    page.wait_for_selector("#p-score-table .status.ok")
    # Norm type and tau stay consistent with the grid, even with tau cleared in between.
    cn_bits = "document.querySelectorAll('#p-norm-grid table')[0].querySelectorAll('td').length"
    page.locator("#p-norm-type").get_by_text("Ternary").click()
    page.wait_for_function(f"{cn_bits} === 18")
    set_text(page, "#p-norm-tau", "")
    page.wait_for_selector("#p-norm-status .status.error")
    page.locator("#p-norm-type").get_by_text("Binary").click()
    page.wait_for_function(f"{cn_bits} === 8")
    page.locator("#p-norm-type").get_by_text("Ternary").click()
    page.wait_for_function(f"{cn_bits} === 18")
    assert page.input_value("#p-norm-tau") == "2"
    choose(page, "p-norm-preset", "Lose a score for not helping someone with a score")
    page.wait_for_function("document.querySelector('#p-barcode').value === 'CN:00000-00 NC:00000000'")
    set_text(page, "#p-eps", "0.1")
    set_text(page, "#p-eps-up", "0.4")
    set_text(page, "#p-eps-down", "0.5")
    page.wait_for_selector("#p-noise-status .status.error")
    set_text(page, "#p-eps-down", "0.6")
    page.wait_for_function("document.querySelector('#p-noise-status').innerText === ''")
    page.click(".advanced summary")
    set_text(page, "#p-maxiter", "300")
    set_text(page, "#p-resolution", "4")
    set_text(page, "#p-tol", "1e-5")
    shot(page, "2_parameters")
    page.click("#p-run")

    # Page 3: results.
    page.wait_for_selector("#r-body:not([style*='none'])", timeout=600_000)
    page.wait_for_selector("#r-simplex .main-svg")
    page.wait_for_selector("#r-hist .main-svg")
    # Nothing is selected and nothing has been evolved yet.
    assert "Click a point" in page.inner_text("#r-summary")
    assert page.is_disabled("#r-evolve") and not page.is_visible("#r-play")
    assert page.locator("input[id*='r-share']").first.input_value() == ""
    shot(page, "3_results")
    before = page.inner_text("#r-summary")
    set_text(page, "input[id*='r-share'] >> nth=0", "0.6")
    set_text(page, "input[id*='r-share'] >> nth=1", "0.2")
    set_text(page, "input[id*='r-share'] >> nth=2", "0.2")
    page.click("#r-show")
    page.wait_for_function(f"document.querySelector('#r-summary').innerText !== {before!r}", timeout=120_000)
    assert "0.6000" in page.inner_text("#r-summary")
    # On the boundary, the absent strategy leaves the table, histogram and matrices.
    set_text(page, "input[id*='r-share'] >> nth=0", "0.5")
    set_text(page, "input[id*='r-share'] >> nth=1", "0.5")
    set_text(page, "input[id*='r-share'] >> nth=2", "0")
    page.click("#r-show")
    page.wait_for_function("!document.querySelector('#r-summary').innerText.includes('S3')", timeout=120_000)
    assert "S3" not in page.inner_text("#r-payoff") and "S3" not in page.inner_text("#r-hist")
    shot(page, "3b_boundary")
    before = page.inner_text("#r-summary")
    click_plot(page, "#r-simplex .nsewdrag", 0.5, 0.55)                  # inside the triangle
    page.wait_for_function(f"document.querySelector('#r-summary').innerText !== {before!r}", timeout=120_000)
    assert page.locator("input[id*='r-share'] >> nth=0").input_value() != ""     # a click fills in its shares
    page.wait_for_function("!document.querySelector('#r-evolve').disabled")
    assert not page.is_visible("#r-play")
    # Evolve from the clicked point: the step size and number of steps are under Advanced, next to the button.
    assert page.get_by_role("button", name="Evolve from point").is_visible()
    assert not page.is_visible("#r-steps")
    page.click(".advanced summary")
    set_text(page, "#r-steps", "6")
    set_text(page, "#r-dt", "1")
    page.click("#r-evolve")
    page.wait_for_selector("#r-play", state="visible", timeout=600_000)
    page.wait_for_function("document.querySelector('#r-evolve-status').innerText.startsWith('Evolved')")
    page.click("#r-play")
    time.sleep(1.5)
    assert "t =" in page.inner_text("#r-time")
    with page.expect_download() as download:
        page.click("#r-download-data")
    assert download.value.suggested_filename == "replicator_results.zip"
    shot(page, "4_after_play")

    # The strategies are on the results page, in their colours and with the labels the plots use.
    chips = page.locator(".strategy-card .chip")
    assert chips.all_inner_texts() == ["S1", "S2", "S3"]
    assert "AllD-Disc" in page.inner_text(".strategy-card >> nth=0") and "Ternary" in page.inner_text(".strategy-card >> nth=2")
    page.get_by_text("Show the p(help) surfaces").click()
    page.wait_for_selector(".strategy-card .mini .main-svg >> nth=2")
    shot(page, "4b_strategies")
    # The first row (strategies and norm) can be folded away.
    page.click("summary:has-text('Strategies and score update norm')")
    page.wait_for_selector(".strategy-card >> nth=0", state="hidden")
    assert not page.is_visible("#r-norm")
    page.click("summary:has-text('Strategies and score update norm')")
    page.wait_for_selector(".strategy-card >> nth=0", state="visible")
    # Rows: strategies and norm side by side; simplex and score distribution side by side below them;
    # then the two matrices and the box with the selected point.
    box = lambda selector: page.locator(selector).bounding_box()
    strategies, norm = box(".card:has(#r-strategies)"), box(".card:has(#r-norm)")
    simplex, hist = box(".card:has(#r-simplex)"), box(".card:has(#r-hist)")
    payoff, helping, chosen = box(".card:has(#r-payoff)"), box(".card:has(#r-help)"), box(".card:has(#r-evolve)")
    assert abs(strategies["y"] - norm["y"]) < 2 and strategies["x"] < norm["x"]
    assert abs(simplex["y"] - hist["y"]) < 2 and simplex["x"] < hist["x"]
    assert simplex["y"] > strategies["y"] + strategies["height"] - 2
    assert abs(payoff["y"] - helping["y"]) < 2 and abs(payoff["y"] - chosen["y"]) < 2
    assert payoff["x"] < helping["x"] < chosen["x"] and payoff["y"] > simplex["y"] + min(simplex["height"], hist["height"]) - 2

    # Back to the parameters and on to the same results again, by the button and by the steps at the top.
    page.click("#r-back")
    page.wait_for_selector("#p-to-results", state="visible")
    assert page.inner_text("#p-results-note") == ""                  # nothing changed since the run
    set_text(page, "#p-eps", "0.2")
    page.wait_for_function("document.querySelector('#p-results-note').innerText.includes('last run')")
    set_text(page, "#p-eps", "0.1")
    page.wait_for_function("document.querySelector('#p-results-note').innerText === ''")
    page.click("#p-to-results")
    page.wait_for_selector("#r-body:not([style*='none'])", timeout=60_000)
    page.wait_for_selector("#r-simplex .main-svg")
    assert chips.all_inner_texts() == ["S1", "S2", "S3"]
    page.click("a.step:has-text('Parameters')")
    page.wait_for_selector("#p-to-results", state="visible")
    page.click("a.step:has-text('Results')")
    page.wait_for_selector("#r-body:not([style*='none'])", timeout=60_000)
    page.click("a.step:has-text('Strategies')")
    page.wait_for_selector("#s-preview .main-svg")
    page.wait_for_selector(".selected-item >> nth=2")
    assert page.locator(".selected-item").count() == 3
    shot(page, "4c_back_and_forth")
    assert not page.errors, page.errors


def test_two_strategies_from_settings_file(page, server, tmp_path):
    """Load a settings file, run, click the line and play."""
    import json

    settings = {
        "version": 1,
        "strategies": [
            {"type": "binary", "name": "Disc-Disc", "P": [[0, 0], [1, 1]], "tau": None, "phi": None},
            {"type": "phi", "name": "phi-Disc", "P": None, "tau": None, "phi": 1.0},
        ],
        "shares": [0.8, 0.2],
        "scores": [{"score": 0, "mass": 0.5}, {"score": 1, "mass": 0.5}],
        "norm": {"type": "binary", "tau": None, "CN": "00000-0+", "NC": "00000000"},
        "noise": {"eps": 0.1, "eps_up": 0.3, "eps_down": 0.7},
        "p_can": 0.5, "benefit": 2, "cost": 1, "dt": 1.0, "n_steps": 15, "max_iter": 300, "tol": 1e-5,
        "resolution": 5,
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    try:
        page.goto(server)
        page.wait_for_selector("#s-upload input[type=file]", state="attached")
        page.set_input_files("#s-upload input[type=file]", str(path))
        page.wait_for_selector("#p-norm-grid table")
        page.wait_for_function("document.querySelector('#p-barcode').value === 'CN:00000-0+ NC:00000000'")
        page.click("#p-run")
        page.wait_for_selector("#r-body:not([style*='none'])", timeout=600_000)
        page.wait_for_selector("#r-simplex .main-svg")
        assert "Click a point" in page.inner_text("#r-summary")
        before = page.inner_text("#r-summary")
        click_plot(page, "#r-simplex .nsewdrag >> nth=-1", 0.3, 0.5)    # the line panel
        page.wait_for_function("document.querySelector('#r-time').innerText === 'selected point'", timeout=120_000)
        assert page.inner_text("#r-summary") != before
        page.wait_for_function("!document.querySelector('#r-evolve').disabled")
        page.click(".advanced summary")
        set_text(page, "#r-steps", "5")
        page.click("#r-evolve")
        page.wait_for_selector("#r-play", state="visible", timeout=600_000)
        page.wait_for_function("document.querySelector('#r-time').innerText.startsWith('t =')")
        shot(page, "5_two_strategies")
        assert not page.errors, page.errors
    except Exception:
        shot(page, "failure_two")
        dump_threads()
        raise


def test_save_and_load_results(page, server, tmp_path, monkeypatch):
    """Save a finished result under a name, forget everything, and load it back without recomputing."""
    import json

    monkeypatch.setenv("DONATION_GAME_SAVES", str(tmp_path))
    settings = {
        "version": 1,
        "strategies": [
            {"type": "binary", "name": "Disc-Disc", "P": [[0, 0], [1, 1]], "tau": None, "phi": None},
            {"type": "phi", "name": "phi-Disc", "P": None, "tau": None, "phi": 1.0},
        ],
        "shares": [0.8, 0.2],
        "scores": [{"score": 0, "mass": 0.5}, {"score": 1, "mass": 0.5}],
        "norm": {"type": "binary", "tau": None, "CN": "00000-0+", "NC": "00000000"},
        "noise": {"eps": 0.1, "eps_up": 0.3, "eps_down": 0.7},
        "p_can": 0.5, "benefit": 2, "cost": 1, "dt": 1.0, "n_steps": 10, "max_iter": 200, "tol": 1e-5,
        "resolution": 4,
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    try:
        page.goto(server)
        page.wait_for_selector("#s-upload input[type=file]", state="attached")
        page.set_input_files("#s-upload input[type=file]", str(path))
        page.wait_for_selector("#p-norm-grid table")
        page.click("#p-run")
        page.wait_for_selector("#r-body:not([style*='none'])", timeout=600_000)
        page.wait_for_selector("#r-simplex .main-svg")
        # The saves are in a pop-up behind a button, hidden until it is clicked.
        assert not page.is_visible("#r-save")
        page.click("#r-saves-open")
        page.wait_for_selector("#r-save", state="visible")
        # A name is needed, and a taken name is refused.
        page.click("#r-save")
        page.wait_for_selector("#r-save-status .status.error")
        set_text(page, "#r-save-name", "my run")
        page.click("#r-save")
        page.wait_for_selector("#r-save-status .status.ok")
        assert (tmp_path / "my_run.sqlite").exists()
        set_text(page, "#r-save-name", "my run")
        page.click("#r-save")
        page.wait_for_function("document.querySelector('#r-save-status').innerText.includes('already')")
        # The save is chosen in the list, so it can be downloaded as a file (to upload again) or as CSVs.
        with page.expect_download() as download:
            page.click("#r-download-file")
        assert download.value.suggested_filename == "my_run.sqlite"
        download.value.save_as(tmp_path / "downloaded.sqlite")
        with page.expect_download() as download:
            page.click("#r-download-csv")
        assert download.value.suggested_filename == "my_run_csv.zip"

        # Forget the session, then load the result back.
        page.evaluate("sessionStorage.clear()")
        page.goto(server + "/results")
        page.wait_for_selector("text=Nothing has been run yet")
        page.click("#r-saves-open")
        page.wait_for_selector("#r-load", state="visible")
        page.click("#r-load")
        page.wait_for_selector("#r-save-status .status.error")                      # nothing chosen
        page.locator("#r-load-choice").click()
        page.get_by_role("option", name=re.compile("^my run")).last.click()
        page.click("#r-load")
        page.wait_for_selector("#r-body:not([style*='none'])", timeout=60_000)
        page.wait_for_selector("#r-simplex .main-svg")
        page.wait_for_selector("#r-hist .main-svg")
        page.wait_for_selector("#r-saves-modal", state="hidden")                  # a load closes the pop-up
        assert page.locator(".strategy-card .chip").all_inner_texts() == ["S1", "S2"]
        assert "CN:00000-0+" in page.inner_text("#r-norm")
        # Its settings are on the parameters page again.
        page.click("a.step:has-text('Parameters')")
        page.wait_for_function("document.querySelector('#p-barcode') && "
                               "document.querySelector('#p-barcode').value === 'CN:00000-0+ NC:00000000'")
        # Rename it (a name is needed), then delete it.
        page.click("a.step:has-text('Results')")
        page.click("#r-saves-open")
        page.wait_for_selector("#r-load-choice", state="visible")
        page.locator("#r-load-choice").click()
        page.get_by_role("option", name=re.compile("^my run")).last.click()
        page.click("#r-rename")
        page.wait_for_selector("#r-save-status .status.error")
        set_text(page, "#r-rename-name", "renamed run")
        page.click("#r-rename")
        page.wait_for_function("document.querySelector('#r-save-status').innerText.includes('Renamed')")
        assert (tmp_path / "renamed_run.sqlite").exists() and not (tmp_path / "my_run.sqlite").exists()
        page.locator("#r-load-choice").click()
        page.get_by_role("option", name=re.compile("^renamed run")).last.click()
        page.once("dialog", lambda dialog: dialog.accept())
        page.click(".modal-panel h3")                          # close the list if it is still open
        page.click("#r-delete")
        page.wait_for_function("document.querySelector('#r-save-status').innerText.includes('Deleted')")
        assert not (tmp_path / "renamed_run.sqlite").exists()
        # Upload the downloaded file: it comes back under its own name (a taken name is refused; see test_saves.py).
        page.set_input_files("#r-upload input[type=file]", str(tmp_path / "downloaded.sqlite"))
        page.wait_for_function("document.querySelector('#r-save-status').innerText.includes('Added')")
        assert (tmp_path / "my_run.sqlite").exists()
        shot(page, "6_saves")
        assert not page.errors, page.errors
    except Exception:
        shot(page, "failure_saves")
        dump_threads()
        raise

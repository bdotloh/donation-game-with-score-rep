"""Click through the GUI in a real browser (Chromium via Playwright).

Run with:  python -m pytest -m e2e tests/test_gui_e2e.py
Screenshots go to $GUI_SCREENSHOTS (default: a temporary directory).
"""

import glob
import os
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


def shot(page, name):
    page.screenshot(path=os.path.join(page.shots, f"{name}.png"), full_page=True)


def choose(page, dropdown_id, option_text):
    page.click(f"#{dropdown_id}")
    page.get_by_role("option", name=option_text, exact=True).click()


def click_plot(page, selector, fx, fy):
    """Hover, then click, at a fraction (fx, fy) of a plot area (Plotly clicks the hovered point)."""
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
        raise


def _full_flow(page):
    # Page 1: strategies.
    page.wait_for_selector("#s-preview .main-svg")
    choose(page, "s-preset", "AllD-Disc")
    page.wait_for_function("document.querySelector('#s-name').value === 'AllD-Disc'")
    page.click("#s-add")
    choose(page, "s-preset", "Defectors")
    page.wait_for_function("document.querySelector('#s-name').value === 'Defectors'")
    page.click("#s-add")
    choose(page, "s-type", "Ternary (score 0 / 1 to τ / above τ)")
    page.wait_for_selector("#s-tau-row:not([style*='none'])")
    page.wait_for_selector(".cell input >> nth=8")
    assert page.locator(".cell input").count() == 9
    for k, value in enumerate(["0", "0", "0", "0", "1", "0.5", "0", "1", "1"]):
        set_text(page, f".cell input >> nth={k}", value)
    set_text(page, "#s-name", "Ternary")
    page.wait_for_function("!document.querySelector('#s-add').disabled")
    page.click("#s-add")
    page.wait_for_selector(".selected-item >> nth=2")
    assert page.locator("#s-add").is_disabled()           # at most 3
    shot(page, "1_strategies")
    page.click("#s-next")

    # Page 2: parameters.
    page.wait_for_selector("#p-norm-grid table")
    for k in range(3):
        set_text(page, f"input[id*='p-share'] >> nth={k}", "33.33")
    page.wait_for_selector("#p-shares-status .status.ok")
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
    choose(page, "p-norm-preset", "Lose a score for not helping someone with a score")
    page.wait_for_function("document.querySelector('#p-barcode').value === 'CN:00000-00 NC:00000000'")
    set_text(page, "#p-eps", "0.1")
    set_text(page, "#p-eps-up", "0.4")
    set_text(page, "#p-eps-down", "0.5")
    page.wait_for_selector("#p-noise-status .status.error")
    set_text(page, "#p-eps-down", "0.6")
    page.wait_for_function("document.querySelector('#p-noise-status').innerText === ''")
    page.click(".advanced summary")
    set_text(page, "#p-steps", "15")
    set_text(page, "#p-maxiter", "300")
    set_text(page, "#p-resolution", "4")
    set_text(page, "#p-tol", "1e-5")
    shot(page, "2_parameters")
    page.click("#p-run")

    # Page 3: results.
    page.wait_for_selector("#r-body:not([style*='none'])", timeout=600_000)
    page.wait_for_selector("#r-simplex .main-svg")
    page.wait_for_selector("#r-hist .main-svg")
    shot(page, "3_results")
    before = page.inner_text("#r-summary")
    set_text(page, "input[id*='r-share'] >> nth=0", "0.6")
    set_text(page, "input[id*='r-share'] >> nth=1", "0.2")
    set_text(page, "input[id*='r-share'] >> nth=2", "0.2")
    page.click("#r-show")
    page.wait_for_function(f"document.querySelector('#r-summary').innerText !== {before!r}", timeout=120_000)
    assert "0.6000" in page.inner_text("#r-summary")
    before = page.inner_text("#r-summary")
    click_plot(page, "#r-simplex .nsewdrag", 0.5, 0.55)                  # inside the triangle
    page.wait_for_function(f"document.querySelector('#r-summary').innerText !== {before!r}", timeout=120_000)
    page.click("#r-play")
    time.sleep(1.5)
    assert "t =" in page.inner_text("#r-time")
    with page.expect_download() as download:
        page.click("#r-download-data")
    assert download.value.suggested_filename == "replicator_results.zip"
    shot(page, "4_after_play")
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
        assert page.input_value("#p-barcode") == "CN:00000-0+ NC:00000000"
        page.click("#p-run")
        page.wait_for_selector("#r-body:not([style*='none'])", timeout=600_000)
        page.wait_for_selector("#r-simplex .main-svg")
        before = page.inner_text("#r-summary")
        click_plot(page, "#r-simplex .nsewdrag >> nth=-1", 0.3, 0.5)    # the line panel
        page.wait_for_function(f"document.querySelector('#r-summary').innerText !== {before!r}", timeout=120_000)
        assert "selected point" in page.inner_text("#r-time")
        shot(page, "5_two_strategies")
        assert not page.errors, page.errors
    except Exception:
        shot(page, "failure_two")
        raise

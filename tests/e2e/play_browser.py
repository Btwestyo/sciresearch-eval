"""Browser end-to-end check of the play app: plays one practice task like a person would.

    uv run --with playwright python tests/e2e/play_browser.py [--headed]

Uses the installed Chrome (channel="chrome"); screenshots go to demo/play_screens/. Exits non-zero on any
page error. Saves its episodes under results/human_test/ (never the real results/human/).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT, TASK = 8799, "task_9211954ed3"
SHOTS = Path("demo/play_screens")


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "sciresearch_eval.play.server", "--port", str(PORT), "--out", "results/human_test"])
    errors = []
    try:
        time.sleep(1.5)
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome", headless="--headed" not in sys.argv)
            page = b.new_page(viewport={"width": 1500, "height": 1100})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("dialog", lambda d: d.accept())
            page.goto(f"http://127.0.0.1:{PORT}/")
            page.fill("#pid", "e2e-test"); page.select_option("#mode", "practice"); page.click("#load")
            page.wait_for_selector(f"button[data-t='{TASK}']"); page.screenshot(path=SHOTS / "1_tasks.png")
            page.click(f"button[data-t='{TASK}']"); page.wait_for_selector("#play:not(.hidden)")
            page.screenshot(path=SHOTS / "2_start.png", full_page=True)
            # a gentle experiment with an inside thermometer
            page.click("#ex-qbtn button[data-f='0.25']"); page.check("#ex-sens input[value='x=0.50L']")
            page.click("#ex-go"); page.wait_for_selector("#srcs button[data-i='d1']")
            # fit d1
            page.click("#tabs button[data-t='fit']")
            page.uncheck("#fit-data input[value='d0']"); page.check("#fit-data input[value='d1']")
            page.click("#fit-go"); page.wait_for_function("document.querySelector('#fitres').innerText.includes('f0')", timeout=300000)
            page.screenshot(path=SHOTS / "3_fit.png", full_page=True)
            # simulate the hidden question from that fit
            page.click("#tabs button[data-t='sim']"); page.select_option("#sim-exp", "hidden"); page.click("#sim-go")
            page.wait_for_selector("#srcs button[data-i='s0']", timeout=300000)
            page.screenshot(path=SHOTS / "4_sim.png", full_page=True)
            # a diagnostic
            page.click("#tabs button[data-t='diag']"); page.select_option("#dg-op", "window_refit"); page.click("#dg-go")
            page.wait_for_function("document.querySelector('#dg-out').innerText.includes('early')", timeout=300000)
            # answer
            page.click("#tabs button[data-t='ans']"); page.click("#ans-prefill")
            for prm in ("k", "rho_cp", "h"):
                e = float(page.input_value(f"#ans-e-{prm}"))
                page.fill(f"#ans-l-{prm}", str(e * 0.85)); page.fill(f"#ans-h-{prm}", str(e * 1.15))
            last = page.evaluate("Object.values(S.sources).filter(s => s.hidden).pop().readings.rear.slice(-1)[0] - 273.15")
            page.fill("#ans-e-target", f"{last:.2f}"); page.fill("#ans-l-target", f"{last - 3:.2f}"); page.fill("#ans-h-target", f"{last + 3:.2f}")
            page.fill("#ans-concl", "browser end-to-end check")
            page.click("#ans-go"); page.wait_for_selector("#done:not(.hidden)", timeout=120000)
            page.screenshot(path=SHOTS / "5_reveal.png", full_page=True)
            b.close()
    finally:
        srv.terminate()
    if errors:
        print("PAGE ERRORS:", *errors, sep="\n  "); sys.exit(1)
    print("ok: played", TASK, "->", sorted(str(p) for p in SHOTS.glob("*.png")))


if __name__ == "__main__":
    main()

"""Browser checks for the T6 candidate-model controls, the T5 solver settings, and baseline mode (no reveal).

    uv run --with playwright python tests/e2e/play_browser_extra.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8798
SHOTS = Path("demo/play_screens")


def first(root: str, split: str) -> str:
    return json.loads(Path(f"{root}/{split}/manifest.json").read_text())["tasks"][0]


def main() -> None:
    srv = subprocess.Popen([sys.executable, "-m", "sciresearch_eval.play.server", "--port", str(PORT), "--out", "results/human_test"])
    errors = []
    try:
        time.sleep(1.5)
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome")
            page = b.new_page(viewport={"width": 1500, "height": 1100})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("dialog", lambda d: d.accept())

            def open_task(mode, task):
                page.goto(f"http://127.0.0.1:{PORT}/"); page.fill("#pid", "e2e-test"); page.select_option("#mode", mode)
                page.click("#load"); page.wait_for_selector(f"button[data-t='{task}']"); page.click(f"button[data-t='{task}']")
                page.wait_for_selector("#play:not(.hidden)")

            # T6: fit a candidate model, then simulate with it
            open_task("practice", first("tasks_t6", "dev"))
            page.click("#tabs button[data-t='fit']"); page.select_option("#fit-model", "slab1_rad")
            assert page.locator("#fit-params input[data-p='emissivity']").count() == 1
            page.click("#fit-go"); page.wait_for_function("document.querySelector('#fitres').innerText.includes('slab1_rad')", timeout=300000)
            page.click("#tabs button[data-t='sim']"); page.select_option("#sim-model", "slab1_rad"); page.select_option("#sim-exp", "d0")
            page.click("#sim-go"); page.wait_for_selector("#srcs button[data-i='s0']", timeout=300000)
            page.screenshot(path=SHOTS / "6_t6_candidate.png", full_page=True)

            # T5: refit with a finer custom mesh (solver settings panel)
            open_task("practice", first("tasks_t5", "dev"))
            page.click("#tabs button[data-t='fit']"); page.click("#tab-fit details summary")
            cells = page.locator("#fit-solver input[data-k='cells_per_layer']")
            cells.fill(str(int(float(cells.input_value())) * 4))
            page.click("#fit-go"); page.wait_for_function("document.querySelector('#log').innerText.includes('custom solver')", timeout=300000)
            page.screenshot(path=SHOTS / "7_t5_solver.png", full_page=True)

            # Baseline: submit without revealing anything
            page.goto(f"http://127.0.0.1:{PORT}/"); page.fill("#pid", "e2e-test"); page.select_option("#mode", "baseline"); page.click("#load")
            page.click("#nextTask"); page.wait_for_selector("#play:not(.hidden)")
            page.click("#tabs button[data-t='fit']"); page.click("#fit-go")
            page.wait_for_function("document.querySelector('#fitres').innerText.includes('f0')", timeout=300000)
            page.click("#tabs button[data-t='ans']"); page.click("#ans-prefill")
            for prm in json.loads(page.evaluate("JSON.stringify(S.v.supplied_model.free_parameters)")):
                e = float(page.input_value(f"#ans-e-{prm}")); page.fill(f"#ans-l-{prm}", str(e * 0.5)); page.fill(f"#ans-h-{prm}", str(e * 2))
            page.fill("#ans-e-target", "100"); page.fill("#ans-l-target", "50"); page.fill("#ans-h-target", "150")
            page.click("#ans-go"); page.wait_for_selector("#done:not(.hidden)")
            text = page.inner_text("#donecard")
            assert "recorded" in text.lower() and "truth" not in text.lower() and "score" not in text.lower(), text
            page.screenshot(path=SHOTS / "8_baseline_done.png")
            b.close()
    finally:
        srv.terminate()
    if errors:
        print("PAGE ERRORS:", *errors, sep="\n  "); sys.exit(1)
    print("ok: T6 candidate fit + sim, T5 custom solver, baseline submit without reveal")


if __name__ == "__main__":
    main()

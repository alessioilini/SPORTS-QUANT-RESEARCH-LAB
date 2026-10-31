"""Dashboard: every section renders without exceptions; the server starts and answers health checks."""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "dashboard" / "app.py")
SECTIONS = ["1. Prematch Analytics", "2. Live Match Analytics", "3. Probability Models", "4. Forecast History",
            "5. Backtesting", "6. Virtual Bankroll", "7. Model Performance", "8. Data Quality"]


@pytest.mark.historical
@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize("sport", ["football", "tennis", "hockey", "basketball"])
def test_every_section_renders(section, sport):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["section"] = section
    at.session_state["sport"] = sport
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(section.split(". ")[1].split()[0] in h.value for h in at.header)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_streamlit_server_starts_and_is_healthy():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", APP, "--server.headless", "true",
                             "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
    try:
        ok = False
        for _ in range(60):
            try:
                r = requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=2,
                                 proxies={"http": None, "https": None})
                if r.status_code == 200 and r.text.strip() == "ok":
                    ok = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        assert ok, "streamlit health endpoint did not answer"
        page = requests.get(f"http://127.0.0.1:{port}/", timeout=5, proxies={"http": None, "https": None})
        assert page.status_code == 200
    finally:
        proc.terminate()
        proc.wait(timeout=20)

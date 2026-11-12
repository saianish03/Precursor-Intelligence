"""Execute one experiment notebook with a RAM guard and a machine-wide lock (one run at a time).

    uv run --no-sync python run_nb.py exp3_lightgbm_goss_efb.ipynb --smoke   # small sample, output -> outputs/smoke/
    uv run --no-sync python run_nb.py exp3_lightgbm_goss_efb.ipynb           # full data, calib gap = horizon
    EXP_CALIB_GAP=0 uv run --no-sync python run_nb.py exp3_...ipynb           # original gold split, executed in place

- Calib gap (EXP_CALIB_GAP, default h): see exp_common. Non-zero gaps write the executed notebook to
  outputs/executed_gap_<setting>/ and results to outputs/results_gap_<setting>/.

- Lock: waits (up to 60 min) for `.run.lock` so parallel agents never run two notebooks at once.
- RAM guard: needs >= MIN_START_GB free to start; kills the run if free RAM stays < KILL_GB for 10 s.
- Kernel: `precursor-exp` (the repo .venv).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import free_ram_gb  # noqa: E402

HERE = Path(__file__).resolve().parent
LOCK = HERE / ".run.lock"
MIN_START_GB = {"smoke": 1.0, "full": 2.0}
KILL_GB = 0.6


def acquire_lock(name: str, wait_s: int = 3600) -> None:
    t0 = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{name} pid={os.getpid()} {time.strftime('%H:%M:%S')}".encode())
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - t0 > wait_s:
                sys.exit(f"lock busy for {wait_s}s: {LOCK.read_text(encoding='utf-8', errors='replace')}")
            time.sleep(10)


def main() -> int:
    nb = HERE / sys.argv[1]
    mode = "smoke" if "--smoke" in sys.argv else "full"
    acquire_lock(nb.name)
    try:
        while free_ram_gb() < MIN_START_GB[mode]:
            print(f"waiting for RAM: {free_ram_gb():.2f} GB free < {MIN_START_GB[mode]} GB", flush=True)
            time.sleep(30)
        env = dict(os.environ, EXP_SMOKE="1" if mode == "smoke" else "0", PYTHONIOENCODING="utf-8")
        gap = os.environ.get("EXP_CALIB_GAP", "h").strip().lower()
        # gap 0 = the original gold split, executed in place; any other gap keeps its executed copy separate
        if mode == "smoke":
            out_args = ["--output-dir", str(HERE / "outputs" / "smoke")]
        elif gap == "0":
            out_args = ["--inplace"]
        else:
            out_args = ["--output-dir", str(HERE / "outputs" / f"executed_gap_{gap}")]
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
               "--ExecutePreprocessor.kernel_name=precursor-exp", "--ExecutePreprocessor.timeout=-1",
               *out_args, str(nb)]
        print(f"[{mode}] {nb.name} start, {free_ram_gb():.2f} GB free", flush=True)
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=HERE, env=env)
        low_since = None
        while p.poll() is None:
            time.sleep(2)
            if free_ram_gb() < KILL_GB:
                low_since = low_since or time.time()
                if time.time() - low_since > 10:
                    p.kill()
                    print(f"KILLED: free RAM < {KILL_GB} GB for 10 s", flush=True)
                    return 3
            else:
                low_since = None
        print(f"[{mode}] {nb.name} exit {p.returncode} after {time.time() - t0:.0f}s", flush=True)
        return p.returncode
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())

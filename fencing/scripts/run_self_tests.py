"""Run every built-in self-test and smoke test; exit non-zero if any fails."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TIMEOUT = 600   # s per test; the slowest loads YOLO and trains two epochs on noise

SELF_TEST_SCRIPTS = ("demo_video", "bout_timeline", "bout_stats", "check_touches",
                     "find_halts", "find_lamps", "read_scoreboard", "sweep_parry_promote",
                     "exp_parry_push", "exp_phrase", "blade_energy")
# (name, interpreter arguments), all run from the project root
TESTS = ([("src/labels", ["-m", "src.labels"]),
          ("src/utils", ["src/utils.py"]),
          ("src/action_model", ["src/action_model.py"]),
          ("src/person_detector", ["src/person_detector.py"]),
          ("src/blade_detector", ["src/blade_detector.py"])]
         + [(f"scripts/{n}", [f"scripts/{n}.py", "--self-test"]) for n in SELF_TEST_SCRIPTS])


def run(name: str, args: list[str]) -> tuple[bool, float, str]:
    """(passed, seconds, last lines of output) for one test."""
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, *args], cwd=PROJECT, capture_output=True,
                           text=True, errors="replace", timeout=TIMEOUT)
        ok, out = r.returncode == 0, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        ok, out = False, f"timed out after {TIMEOUT}s"
    return ok, time.time() - t0, "\n".join(out.strip().splitlines()[-12:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", metavar="TEXT", help="only tests whose name contains TEXT")
    a = ap.parse_args()
    tests = [t for t in TESTS if not a.k or a.k in t[0]]
    failed = []
    for name, args in tests:
        ok, secs, tail = run(name, args)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<28}{secs:6.1f}s", flush=True)
        if not ok:
            failed.append(name)
            print("\n".join(f"        {line}" for line in tail.splitlines()))
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

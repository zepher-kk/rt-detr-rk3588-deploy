#!/usr/bin/env python3
"""Run one command and report wall time, CPU time, utilization, and peak RSS."""

import argparse
import json
import resource
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("missing command")

    started = time.perf_counter()
    result = subprocess.run(args.command, check=False)
    wall_s = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = usage.ru_utime + usage.ru_stime
    report = {
        "returncode": result.returncode,
        "wall_s": round(wall_s, 6),
        "user_s": round(usage.ru_utime, 6),
        "system_s": round(usage.ru_stime, 6),
        "cpu_s": round(cpu_s, 6),
        "cpu_percent": round(100.0 * cpu_s / wall_s, 3) if wall_s else 0.0,
        "max_rss_kb": usage.ru_maxrss,
    }
    print("BENCH_RESULT=" + json.dumps(report, sort_keys=True), flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

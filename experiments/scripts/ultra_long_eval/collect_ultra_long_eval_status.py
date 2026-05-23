from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LOCAL_REPO = Path(r"F:\bishe\GeoExplorer")
LOCAL_OUTPUT_ROOT = LOCAL_REPO / "analysis" / "pipeline_20260521_ultra_long_grid_stress_v3_grid25"
TUNING_DIR = LOCAL_REPO / "tuning"
if str(TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(TUNING_DIR))

from remote_geo import RemoteGeoClient, RemoteSpec, shell_quote


REMOTE_EXP_ROOT = "/root/geoexplorer/ab_experiments/ultra_long_eval_20260521/anchor0624_ultralong_grid_stress"
REMOTE_OUTPUT_ROOT = "/root/geoexplorer/analysis/pipeline_20260521_ultra_long_grid_stress_v3_grid25"
STATUS_PATH = REMOTE_EXP_ROOT + "/monitoring/ultra_long_status_latest.json"
NVIDIA_COMPAT_LIB = "/root/geoexplorer/env/nvidia_535_288/usr/lib/x86_64-linux-gnu"
NVIDIA_COMPAT_SMI = "/root/geoexplorer/env/nvidia_535_288/usr/bin/nvidia-smi"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect ultra-long grid stress evaluation status.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--key-file", default="")
    parser.add_argument("--no-look-for-keys", action="store_true")
    parser.add_argument("--no-agent", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args()

    spec = RemoteSpec(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        key_file=args.key_file,
        look_for_keys=not args.no_look_for_keys and not bool(args.password),
        allow_agent=not args.no_agent and not bool(args.password),
    )
    with RemoteGeoClient(spec) as client:
        status_raw = client.run(f"cat {shell_quote(STATUS_PATH)} 2>/dev/null || true")
        gpu = client.run(
            f"LD_LIBRARY_PATH={shell_quote(NVIDIA_COMPAT_LIB)} "
            f"{shell_quote(NVIDIA_COMPAT_SMI)} --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits"
        )
        proc = client.run(
            "ps -eo pid,ppid,user,stat,etime,cmd | "
            "grep -E 'ultra_long_supervisor.py|paper_geo_evaluator.py' | grep -v grep || true"
        )
        downloaded = False
        if args.download and client.path_exists(REMOTE_OUTPUT_ROOT):
            LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            client.download_dir(REMOTE_OUTPUT_ROOT, str(LOCAL_OUTPUT_ROOT))
            downloaded = True

    status = json.loads(status_raw.stdout) if status_raw.stdout.strip().startswith("{") else None
    if args.brief and status:
        eval_jobs = status.get("eval_jobs", {})
        counts = {}
        if isinstance(eval_jobs, dict):
            for row in eval_jobs.values():
                counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
        payload = {
            "phase": status.get("phase"),
            "timestamp": status.get("timestamp"),
            "mode": status.get("mode"),
            "grids": status.get("grids"),
            "active_eval_processes": status.get("active_eval_processes"),
            "eval_status_counts": counts,
            "output_root": status.get("output_root"),
            "gpu": gpu.stdout,
            "processes": proc.stdout,
            "downloaded": downloaded,
        }
    else:
        payload = {
            "status": status or status_raw.stdout.strip(),
            "gpu": gpu.stdout,
            "processes": proc.stdout,
            "downloaded": downloaded,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

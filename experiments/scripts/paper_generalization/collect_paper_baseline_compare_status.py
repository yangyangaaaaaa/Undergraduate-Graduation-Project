from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LOCAL_REPO = Path(r"F:\bishe\GeoExplorer")
TUNING_DIR = LOCAL_REPO / "tuning"
if str(TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(TUNING_DIR))

from remote_geo import RemoteGeoClient, add_remote_connection_args, remote_spec_from_args, shell_quote


REMOTE_EXP_ROOT = "/root/geoexplorer/ab_experiments/algo_paper_generalization_20260516/anchor0624_factorial_generalization_seed321_480k"
STATUS_PATH = REMOTE_EXP_ROOT + "/monitoring/paper_baseline_compare_status_latest.json"
REMOTE_OUTPUT_ROOT = "/root/geoexplorer/analysis/pipeline_20260516_paper_baseline_compare"
LOCAL_OUTPUT_ROOT = LOCAL_REPO / "analysis" / "pipeline_20260516_paper_baseline_compare"


def summarize_brief(status_payload: dict) -> dict:
    runs = status_payload.get("runs", {})
    counts = {}
    running = []
    failed = []
    for run_name, run in runs.items():
        status = run.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status == "running":
            running.append({"run": run_name, "method": run.get("method"), "benchmark": run.get("benchmark"), "gpu": run.get("gpu"), "pid": run.get("pid")})
        if str(status).startswith("failed"):
            failed.append({"run": run_name, "method": run.get("method"), "benchmark": run.get("benchmark"), "status": status, "returncode": run.get("returncode")})
    return {
        "timestamp": status_payload.get("timestamp"),
        "phase": status_payload.get("phase"),
        "active_eval_processes": status_payload.get("active_eval_processes"),
        "run_counts": counts,
        "running": running,
        "failed": failed,
        "summary_path": status_payload.get("summary_path"),
        "table_path": status_payload.get("table_path"),
        "aggregate_path": status_payload.get("aggregate_path"),
        "blocked_or_not_applicable": status_payload.get("blocked_or_not_applicable"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect paper-aligned baseline comparison status.")
    add_remote_connection_args(parser)
    parser.add_argument("--download", action="store_true", help="Download remote output directory when available.")
    parser.add_argument("--brief", action="store_true", help="Print a compact progress summary instead of full status JSON.")
    args = parser.parse_args()
    spec = remote_spec_from_args(args)
    with RemoteGeoClient(spec) as client:
        status_result = client.run(f"cat {shell_quote(STATUS_PATH)} 2>/dev/null || true")
        gpu = client.run("nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits")
        processes = client.run(
            "ps -eo pid,ppid,user,stat,etime,cmd | "
            "grep -E 'paper_baseline_compare_supervisor.py|paper_baseline_evaluator.py' | "
            "grep -v grep || true"
        )
        downloaded_to = None
        status_payload = json.loads(status_result.stdout) if status_result.stdout.strip().startswith("{") else None
        if args.download and status_payload is not None:
            client.download_dir(REMOTE_OUTPUT_ROOT, str(LOCAL_OUTPUT_ROOT))
            downloaded_to = str(LOCAL_OUTPUT_ROOT)

    payload = {"status": status_payload, "gpu": gpu.stdout, "processes": processes.stdout, "downloaded_to": downloaded_to}
    if args.brief and status_payload is not None:
        payload["brief"] = summarize_brief(status_payload)
        payload.pop("status", None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

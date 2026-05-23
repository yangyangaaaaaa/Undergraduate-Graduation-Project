from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LOCAL_REPO = Path(r"F:\bishe\GeoExplorer")
LOCAL_OUTPUT_ROOT = LOCAL_REPO / "analysis" / "pipeline_20260519_appendix_gate_valdist_dense_followup"
TUNING_DIR = LOCAL_REPO / "tuning"
if str(TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(TUNING_DIR))

from remote_geo import RemoteGeoClient, RemoteSpec, shell_quote


REMOTE_EXP_ROOT = "/root/geoexplorer/ab_experiments/appendix_compare_20260519/anchor0624_gate_valdist_dense_followup_seed321_480k"
REMOTE_OUTPUT_ROOT = "/root/geoexplorer/analysis/pipeline_20260519_appendix_gate_valdist_dense_followup"
STATUS_PATH = REMOTE_EXP_ROOT + "/monitoring/appendix_gate_valdist_status_latest.json"
NVIDIA_COMPAT_LIB = "/root/geoexplorer/env/nvidia_535_288/usr/lib/x86_64-linux-gnu"
NVIDIA_COMPAT_SMI = "/root/geoexplorer/env/nvidia_535_288/usr/bin/nvidia-smi"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect appendix gate/validation-distance follow-up status.")
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
            "grep -E 'appendix_gate_valdist_orchestrator.py|appendix_dataset_param_orchestrator.py|train.py|paper_geo_evaluator.py' | grep -v grep || true"
        )
        downloaded = []
        if args.download and status_raw.stdout.strip():
            LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            for name in [
                "appendix_all_results.json",
                "appendix_long_table.csv",
                "appendix_gate_valdist_sr_table.csv",
                "appendix_gate_valdist_sg_table.csv",
                "appendix_gate_valdist_per_distance.csv",
                "appendix_summary_zh.md",
            ]:
                remote_path = f"{REMOTE_OUTPUT_ROOT}/{name}"
                local_path = LOCAL_OUTPUT_ROOT / name
                if client.path_exists(remote_path):
                    client.download(remote_path, str(local_path))
                    downloaded.append(str(local_path))

    status = json.loads(status_raw.stdout) if status_raw.stdout.strip() else None
    if args.brief and status:
        phase = status.get("phase")
        active_train = status.get("active_train_processes")
        active_eval = status.get("active_eval_processes")
        total_train = status.get("total_train_runs")
        training_runs = status.get("training_runs", {})
        counts = {}
        if isinstance(training_runs, dict):
            for row in training_runs.values():
                counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
        payload = {
            "phase": phase,
            "active_train_processes": active_train,
            "active_eval_processes": active_eval,
            "total_train_runs": total_train,
            "training_status_counts": counts,
            "gpu": gpu.stdout,
            "processes": proc.stdout,
            "downloaded": downloaded,
        }
    else:
        payload = {
            "status": status,
            "gpu": gpu.stdout,
            "processes": proc.stdout,
            "downloaded": downloaded,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LOCAL_REPO = Path(r"F:\bishe\GeoExplorer")
TUNING_DIR = LOCAL_REPO / "tuning"
if str(TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(TUNING_DIR))

from remote_geo import RemoteGeoClient, RemoteSpec, shell_quote


REMOTE_EXP_ROOT = "/root/geoexplorer/ab_experiments/algo_ablation_anchor0624_20260515/anchor_val78_component_ablation_seed321_480k_shared"
STATUS_PATH = REMOTE_EXP_ROOT + "/monitoring/anchor0624_ablation_status_latest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect GeoExplorer anchor0624 factorial ablation status.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--key-file", default="")
    parser.add_argument("--no-look-for-keys", action="store_true")
    parser.add_argument("--no-agent", action="store_true")
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
        status = client.run(f"cat {shell_quote(STATUS_PATH)} 2>/dev/null || true")
        gpu = client.run("nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits")
        proc = client.run(
            "ps -eo pid,ppid,user,stat,etime,cmd | "
            "grep -E 'anchor0624_ablation_orchestrator.py|train.py|geo_shared_eval_one.py' | grep -v grep || true"
        )
    payload = {
        "status": json.loads(status.stdout) if status.stdout.strip() else None,
        "gpu": gpu.stdout,
        "processes": proc.stdout,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

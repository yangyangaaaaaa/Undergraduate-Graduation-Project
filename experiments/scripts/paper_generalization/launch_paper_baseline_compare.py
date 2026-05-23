from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
SERIES_ROOT = EXPERIMENT_ROOT.parent
LOCAL_REPO = Path(r"F:\bishe\GeoExplorer")
TUNING_DIR = LOCAL_REPO / "tuning"
if str(TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(TUNING_DIR))

from remote_geo import RemoteGeoClient, add_remote_connection_args, remote_spec_from_args, shell_quote


REMOTE_EXP_ROOT = f"/root/geoexplorer/ab_experiments/{SERIES_ROOT.name}/{EXPERIMENT_ROOT.name}"
REMOTE_MONITORING = f"{REMOTE_EXP_ROOT}/monitoring"
LOCAL_SUPERVISOR = SCRIPT_DIR / "paper_baseline_compare_supervisor.py"
REMOTE_SUPERVISOR = f"{REMOTE_MONITORING}/paper_baseline_compare_supervisor.py"
REMOTE_STATUS = f"{REMOTE_MONITORING}/paper_baseline_compare_status_latest.json"
REMOTE_PYTHONPATH = "/root/geoexplorer/env/geoexplorer_site:/root/geoexplorer:/root/geoexplorer/GeoExplorer"


def upload_pipeline_files(client: RemoteGeoClient) -> None:
    client.ensure_remote_dir(REMOTE_EXP_ROOT)
    client.upload(str(EXPERIMENT_ROOT / "comparison_manifest.json"), f"{REMOTE_EXP_ROOT}/comparison_manifest.json")
    client.upload_dir(str(SCRIPT_DIR), REMOTE_MONITORING)


def compile_scripts(client: RemoteGeoClient) -> None:
    result = client.run(
        "/usr/bin/python3 -m py_compile "
        + shell_quote(REMOTE_SUPERVISOR)
        + " "
        + shell_quote(f"{REMOTE_MONITORING}/paper_baseline_evaluator.py")
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def active_related_processes(client: RemoteGeoClient) -> list[dict]:
    script = """
import json, os
rows = []
for pid in sorted(p for p in os.listdir('/proc') if p.isdigit()):
    try:
        raw = open(f'/proc/{pid}/cmdline', 'rb').read()
        cwd = os.readlink(f'/proc/{pid}/cwd')
    except Exception:
        continue
    if not raw:
        continue
    parts = [x.decode('utf-8', 'replace') for x in raw.split(b'\\x00') if x]
    joined = ' '.join(parts)
    if any(part.endswith('paper_baseline_compare_supervisor.py') for part in parts):
        rows.append({'pid': int(pid), 'cwd': cwd, 'cmdline': joined})
print(json.dumps(rows, ensure_ascii=False))
"""
    result = client.run("/usr/bin/python3 - <<'PY'\n" + script + "\nPY")
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout or "[]")


def launch_supervisor(client: RemoteGeoClient) -> dict:
    active = active_related_processes(client)
    if active:
        return {"already_running": True, "active": active}
    command = (
        "sh -lc "
        + shell_quote(
            f"cd {shell_quote(REMOTE_MONITORING)}; "
            f"PYTHONPATH={shell_quote(REMOTE_PYTHONPATH)} "
            "nohup /usr/bin/python3 -u paper_baseline_compare_supervisor.py "
            "> paper_baseline_compare_supervisor.stdout.log 2> paper_baseline_compare_supervisor.stderr.log < /dev/null & "
            "echo $! > paper_baseline_compare_supervisor.launch.pid; "
            "sleep 2; cat paper_baseline_compare_supervisor.launch.pid"
        )
    )
    result = client.run(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "failed to launch paper baseline compare")
    return {"already_running": False, "supervisor_pid": result.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload and launch paper-aligned baseline comparison.")
    add_remote_connection_args(parser)
    args = parser.parse_args()
    spec = remote_spec_from_args(args)
    with RemoteGeoClient(spec) as client:
        upload_pipeline_files(client)
        compile_scripts(client)
        launch = launch_supervisor(client)
        status = client.run(f"cat {shell_quote(REMOTE_STATUS)} 2>/dev/null || true")
        gpu = client.run("nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits")
    print(
        json.dumps(
            {
                "remote_experiment_root": REMOTE_EXP_ROOT,
                "remote_monitoring": REMOTE_MONITORING,
                "launch": launch,
                "initial_status": json.loads(status.stdout) if status.stdout.strip().startswith("{") else status.stdout.strip(),
                "gpu": gpu.stdout,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

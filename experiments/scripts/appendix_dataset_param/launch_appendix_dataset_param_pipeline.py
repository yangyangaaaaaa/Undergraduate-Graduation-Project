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

from remote_geo import RemoteGeoClient, RemoteSpec, shell_quote


REMOTE_REPO = "/root/geoexplorer/GeoExplorer"
REMOTE_EXP_ROOT = f"/root/geoexplorer/ab_experiments/{SERIES_ROOT.name}/{EXPERIMENT_ROOT.name}"
REMOTE_MONITORING = f"{REMOTE_EXP_ROOT}/monitoring"
STATUS_PATH = f"{REMOTE_MONITORING}/appendix_dataset_param_status_latest.json"
REMOTE_PYTHONPATH = (
    "/root/geoexplorer/env/geoexplorer_site:"
    "/root/geoexplorer:"
    "/root/geoexplorer/GeoExplorer:"
    "/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle"
)

CODE_FILES = (
    "config.py",
    "train.py",
    "models/ppo.py",
    "models/pretrain_model.py",
    "models/model_falcon.py",
    "models/decision_transformer.py",
    "data_utils/__init__.py",
    "data_utils/sequence.py",
    "utils/__init__.py",
    "utils/get_test_config.py",
    "utils/random_seed.py",
)


def upload_code(client: RemoteGeoClient) -> None:
    for rel in CODE_FILES:
        client.upload(str(LOCAL_REPO / rel), f"{REMOTE_REPO}/{rel}")


def upload_pipeline_files(client: RemoteGeoClient) -> None:
    client.ensure_remote_dir(REMOTE_EXP_ROOT)
    client.upload(str(EXPERIMENT_ROOT / "comparison_manifest.json"), f"{REMOTE_EXP_ROOT}/comparison_manifest.json")
    client.upload_dir(str(SCRIPT_DIR), REMOTE_MONITORING)


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
    if any(part.endswith('appendix_dataset_param_orchestrator.py') for part in parts):
        rows.append({'pid': int(pid), 'cwd': cwd, 'cmdline': joined})
print(json.dumps(rows, ensure_ascii=False))
"""
    result = client.run("/usr/bin/python3 - <<'PY'\n" + script + "\nPY")
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout or "[]")


def launch_orchestrator(client: RemoteGeoClient) -> dict:
    active = active_related_processes(client)
    if active:
        return {"already_running": True, "active": active}
    command = (
        "sh -lc "
        + shell_quote(
            f"cd {shell_quote(REMOTE_MONITORING)}; "
            f"PYTHONPATH={shell_quote(REMOTE_PYTHONPATH)} "
            f"nohup /usr/bin/python3 -u appendix_dataset_param_orchestrator.py "
            f"> appendix_dataset_param_orchestrator.stdout.log 2> appendix_dataset_param_orchestrator.stderr.log < /dev/null & "
            f"echo $! > appendix_dataset_param_orchestrator.launch.pid; "
            f"sleep 2; cat appendix_dataset_param_orchestrator.launch.pid"
        )
    )
    result = client.run(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "failed to launch appendix pipeline")
    return {"already_running": False, "orchestrator_pid": result.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload and launch appendix dataset/parameter comparison pipeline.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--key-file", default="")
    parser.add_argument("--no-look-for-keys", action="store_true")
    parser.add_argument("--no-agent", action="store_true")
    parser.add_argument("--skip-code-upload", action="store_true")
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
        if not args.skip_code_upload:
            upload_code(client)
        upload_pipeline_files(client)
        launch = launch_orchestrator(client)
        status = client.run(f"cat {shell_quote(STATUS_PATH)} 2>/dev/null || true")
    print(
        json.dumps(
            {
                "remote_experiment_root": REMOTE_EXP_ROOT,
                "remote_monitoring": REMOTE_MONITORING,
                "launch": launch,
                "initial_status": status.stdout.strip(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

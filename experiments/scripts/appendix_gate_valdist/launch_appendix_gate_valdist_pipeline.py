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
STATUS_PATH = f"{REMOTE_MONITORING}/appendix_gate_valdist_status_latest.json"
ORCHESTRATOR_SCRIPT = "appendix_gate_valdist_orchestrator.py"
DATASET_PARAM_ORCHESTRATOR_SCRIPT = "appendix_dataset_param_orchestrator.py"
NVIDIA_COMPAT_LIB = "/root/geoexplorer/env/nvidia_535_288/usr/lib/x86_64-linux-gnu"
PARAMETER_PRIORITIES = {
    "parameter",
    "parameters",
    "param",
    "parameter_fullrange",
    "gate_valdist_param",
    "gate_valdist",
    "gate_floor_dense",
    "gate_floor",
    "val_dists_bias",
    "val_dists",
    "validation_distance",
}
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


def active_related_processes(client: RemoteGeoClient, blockers: tuple[str, ...]) -> list[dict]:
    script = """
import json, os
blockers = set(%s)
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
    if any(any(part.endswith(blocker) for blocker in blockers) for part in parts):
        rows.append({'pid': int(pid), 'cwd': cwd, 'cmdline': joined})
print(json.dumps(rows, ensure_ascii=False))
""" % (json.dumps(list(blockers)),)
    result = client.run("/usr/bin/python3 - <<'PY'\n" + script + "\nPY")
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout or "[]")


def blocker_scripts(priority: str) -> tuple[str, ...]:
    normalized = priority.strip().lower()
    blockers = [ORCHESTRATOR_SCRIPT]
    if normalized in PARAMETER_PRIORITIES:
        blockers.append(DATASET_PARAM_ORCHESTRATOR_SCRIPT)
    return tuple(blockers)


def launch_orchestrator(client: RemoteGeoClient, priority: str) -> dict:
    active = active_related_processes(client, blocker_scripts(priority))
    if active:
        return {"already_running": True, "active": active}
    priority_env = f"GEOEXPLORER_PRIORITY_ABLATION_ONLY={shell_quote(priority)} " if priority else ""
    command = (
        "sh -lc "
        + shell_quote(
            f"cd {shell_quote(REMOTE_MONITORING)}; "
            f"PYTHONPATH={shell_quote(REMOTE_PYTHONPATH)} "
            f"LD_LIBRARY_PATH={shell_quote(NVIDIA_COMPAT_LIB)} "
            f"{priority_env}"
            f"nohup /usr/bin/python3 -u {ORCHESTRATOR_SCRIPT} "
            f"> appendix_gate_valdist_orchestrator.stdout.log 2> appendix_gate_valdist_orchestrator.stderr.log < /dev/null & "
            f"echo $! > appendix_gate_valdist_orchestrator.launch.pid; "
            f"sleep 2; cat appendix_gate_valdist_orchestrator.launch.pid"
        )
    )
    result = client.run(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "failed to launch appendix pipeline")
    return {"already_running": False, "orchestrator_pid": result.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload and launch appendix gate/validation-distance follow-up pipeline.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--key-file", default="")
    parser.add_argument("--no-look-for-keys", action="store_true")
    parser.add_argument("--no-agent", action="store_true")
    parser.add_argument("--skip-code-upload", action="store_true")
    parser.add_argument(
        "--priority",
        default="reward_control",
        help=(
            "Subset to launch via GEOEXPLORER_PRIORITY_ABLATION_ONLY. "
            "Use 'parameter_fullrange' for gate_floor_dense + val_dists_bias."
        ),
    )
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
        launch = launch_orchestrator(client, args.priority)
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

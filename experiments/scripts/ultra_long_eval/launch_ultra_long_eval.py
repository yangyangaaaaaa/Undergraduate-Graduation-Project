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
EVALUATOR = (
    LOCAL_REPO
    / "ab_experiments"
    / "appendix_compare_20260519"
    / "anchor0624_gate_valdist_dense_followup_seed321_480k"
    / "monitoring"
    / "paper_geo_evaluator.py"
)

if str(TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(TUNING_DIR))

from remote_geo import RemoteGeoClient, RemoteSpec, shell_quote


REMOTE_EXP_ROOT = f"/root/geoexplorer/ab_experiments/{SERIES_ROOT.name}/{EXPERIMENT_ROOT.name}"
REMOTE_MONITORING = f"{REMOTE_EXP_ROOT}/monitoring"
REMOTE_STATUS = f"{REMOTE_MONITORING}/ultra_long_status_latest.json"
REMOTE_OUTPUT_ROOT = "/root/geoexplorer/analysis/pipeline_20260521_ultra_long_grid_stress_v3_grid25"
NVIDIA_COMPAT_LIB = "/root/geoexplorer/env/nvidia_535_288/usr/lib/x86_64-linux-gnu"
REMOTE_PYTHONPATH = (
    "/root/geoexplorer/env/geoexplorer_site:"
    "/root/geoexplorer:"
    "/root/geoexplorer/GeoExplorer:"
    "/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle"
)


def upload_pipeline_files(client: RemoteGeoClient) -> None:
    client.ensure_remote_dir(REMOTE_MONITORING)
    client.upload(str(EXPERIMENT_ROOT / "README.md"), f"{REMOTE_EXP_ROOT}/README.md")
    client.upload(str(SCRIPT_DIR / "ultra_long_supervisor.py"), f"{REMOTE_MONITORING}/ultra_long_supervisor.py")
    client.upload(str(EVALUATOR), f"{REMOTE_MONITORING}/paper_geo_evaluator.py")


def compile_scripts(client: RemoteGeoClient) -> None:
    result = client.run(
        "/usr/bin/python3 -m py_compile "
        + shell_quote(f"{REMOTE_MONITORING}/ultra_long_supervisor.py")
        + " "
        + shell_quote(f"{REMOTE_MONITORING}/paper_geo_evaluator.py")
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def active_related_processes(client: RemoteGeoClient) -> list[dict]:
    script = """
import json, os
rows = []
targets = {'ultra_long_supervisor.py'}
for pid in sorted(p for p in os.listdir('/proc') if p.isdigit()):
    try:
        raw = open(f'/proc/{pid}/cmdline', 'rb').read()
        cwd = os.readlink(f'/proc/{pid}/cwd')
    except Exception:
        continue
    if not raw:
        continue
    parts = [x.decode('utf-8', 'replace') for x in raw.split(b'\\x00') if x]
    if any(any(part.endswith(target) for target in targets) for part in parts):
        rows.append({'pid': int(pid), 'cwd': cwd, 'cmdline': ' '.join(parts)})
print(json.dumps(rows, ensure_ascii=False))
"""
    result = client.run("/usr/bin/python3 - <<'PY'\n" + script + "\nPY")
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout or "[]")


def launch_supervisor(
    client: RemoteGeoClient,
    mode: str,
    grids: str,
    eval_gpus: str,
    embed_gpu: int,
    force_rebuild: bool,
    max_images: int,
    repeats_per_dist: int,
) -> dict:
    active = active_related_processes(client)
    if active:
        return {"already_running": True, "active": active}

    args = ["--mode", mode, "--eval-gpus", eval_gpus]
    if grids:
        args.extend(["--grids", grids])
    if force_rebuild:
        args.append("--force-rebuild")
    if max_images >= 0:
        args.extend(["--max-images", str(max_images)])
    if repeats_per_dist > 0:
        args.extend(["--repeats-per-dist", str(repeats_per_dist)])

    env_bits = [
        f"PYTHONPATH={shell_quote(REMOTE_PYTHONPATH)}",
        f"LD_LIBRARY_PATH={shell_quote(NVIDIA_COMPAT_LIB)}",
        f"CUDA_VISIBLE_DEVICES={int(embed_gpu)}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
    ]
    command = (
        "sh -lc "
        + shell_quote(
            f"cd {shell_quote(REMOTE_MONITORING)}; "
            + " ".join(env_bits)
            + " nohup /usr/bin/python3 -u ultra_long_supervisor.py "
            + " ".join(shell_quote(item) for item in args)
            + " > ultra_long_supervisor.stdout.log 2> ultra_long_supervisor.stderr.log < /dev/null & "
            + "echo $! > ultra_long_supervisor.launch.pid; "
            + "sleep 2; cat ultra_long_supervisor.launch.pid"
        )
    )
    result = client.run(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "failed to launch ultra-long supervisor")
    return {
        "already_running": False,
        "supervisor_pid": result.stdout.strip(),
        "mode": mode,
        "grids": grids,
        "eval_gpus": eval_gpus,
        "embed_gpu": int(embed_gpu),
        "force_rebuild": bool(force_rebuild),
        "max_images": int(max_images),
        "repeats_per_dist": int(repeats_per_dist),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload and launch the ultra-long grid stress evaluation.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--key-file", default="")
    parser.add_argument("--no-look-for-keys", action="store_true")
    parser.add_argument("--no-agent", action="store_true")
    parser.add_argument("--mode", choices=["smoke", "formal", "full"], default="smoke")
    parser.add_argument("--grids", default="")
    parser.add_argument("--eval-gpus", default="0,1,2")
    parser.add_argument("--embed-gpu", type=int, default=0)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--repeats-per-dist", type=int, default=0)
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
        upload_pipeline_files(client)
        compile_scripts(client)
        launch = launch_supervisor(
            client,
            args.mode,
            args.grids,
            args.eval_gpus,
            args.embed_gpu,
            args.force_rebuild,
            args.max_images,
            args.repeats_per_dist,
        )
        status = client.run(f"cat {shell_quote(REMOTE_STATUS)} 2>/dev/null || true")
        gpu = client.run(
            f"LD_LIBRARY_PATH={shell_quote(NVIDIA_COMPAT_LIB)} "
            "/root/geoexplorer/env/nvidia_535_288/usr/bin/nvidia-smi "
            "--query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits"
        )

    print(
        json.dumps(
            {
                "remote_experiment_root": REMOTE_EXP_ROOT,
                "remote_monitoring": REMOTE_MONITORING,
                "remote_output_root": REMOTE_OUTPUT_ROOT,
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

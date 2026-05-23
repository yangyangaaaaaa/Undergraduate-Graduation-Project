import json
from pathlib import Path


MONUMENTS_METADATA_FILENAMES = [
    "SwissViewMonuments.normalized.json",
    "SwissViewMonuments.json",
]

MONUMENTS_ASSET_ROOT_CANDIDATES = [
    Path("data/swissview/data/SwissViewMonuments"),
    Path("data/swissview/SwissViewMonuments"),
    Path("data/SwissViewMonuments"),
]

_ALIAS_GROUPS = [
    {"Cilion", "Chillon", "CilionCastle"},
    {"Geneve", "Geneva"},
    {"LacBleu", "LacBlue"},
    {"RhineGlacier", "RhoneGlacier"},
]

ALIAS_LOOKUP = {}
for group in _ALIAS_GROUPS:
    ordered = sorted(group)
    for item in ordered:
        ALIAS_LOOKUP[item] = ordered


def get_monuments_metadata_candidates(repo_root):
    base = Path(repo_root) / "data" / "swissview"
    return [base / filename for filename in MONUMENTS_METADATA_FILENAMES]


def choose_monuments_metadata_path(repo_root, override=None):
    if override:
        override_path = Path(override)
        if not override_path.is_absolute():
            override_path = Path(repo_root) / override_path
        return override_path

    for candidate in get_monuments_metadata_candidates(repo_root):
        if candidate.exists():
            return candidate

    return get_monuments_metadata_candidates(repo_root)[-1]


def load_monuments_metadata(repo_root, override=None):
    metadata_path = choose_monuments_metadata_path(repo_root, override=override)
    if not metadata_path.exists():
        return [], metadata_path
    return json.loads(metadata_path.read_text(encoding="utf-8")), metadata_path


def find_monuments_asset_root(repo_root):
    repo_root = Path(repo_root)
    for rel_path in MONUMENTS_ASSET_ROOT_CANDIDATES:
        candidate = repo_root / rel_path
        if candidate.exists():
            return candidate
    return repo_root / MONUMENTS_ASSET_ROOT_CANDIDATES[0]


def _build_name_candidates(filename):
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix.lower()
    if "_" not in stem:
        return [path.name]

    prefix, index_suffix = stem.rsplit("_", 1)
    prefixes = ALIAS_LOOKUP.get(prefix, [prefix])
    if suffix == ".jpg":
        suffixes = [".jpg", ".jpeg"]
    elif suffix == ".jpeg":
        suffixes = [".jpeg", ".jpg"]
    else:
        suffixes = [path.suffix]

    names = []
    seen = set()
    for alias in prefixes:
        for ext in suffixes:
            candidate = f"{alias}_{index_suffix}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                names.append(candidate)
    return names


def resolve_monuments_asset(repo_root, rel_path):
    repo_root = Path(repo_root)
    rel_path = Path(rel_path)

    direct_candidates = [
        repo_root / "data" / "swissview" / rel_path,
        repo_root / rel_path,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate, True

    asset_root = find_monuments_asset_root(repo_root)
    subdir = rel_path.parent.name
    filename = rel_path.name
    if subdir in {"aerial_view", "ground_view"}:
        target_dir = asset_root / subdir

        direct_name = target_dir / filename
        if direct_name.exists():
            return direct_name, True

        for candidate_name in _build_name_candidates(filename):
            candidate = target_dir / candidate_name
            if candidate.exists():
                return candidate, False

        stem = Path(filename).stem
        if "_" in stem:
            _, index_suffix = stem.rsplit("_", 1)
            matches = sorted(target_dir.glob(f"*_{index_suffix}.*"))
            if len(matches) == 1:
                return matches[0], False

    fallback = repo_root / "data" / "swissview" / rel_path
    return fallback, False


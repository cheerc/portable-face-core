"""Corpus manifest schema + loader (Task 9). The manifest lives outside Git."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from facecore.errors import ConfigurationError

CONDITION_ORDER = (
    "age_change",
    "hairstyle",
    "eyewear",
    "hats",
    "pose",
    "lighting",
    "complex_background",
    "masks",
    "blur",
    "occlusion",
)

ROLES = ("enrollment", "target_probe", "non_target_probe")


@dataclass(frozen=True)
class CorpusFile:
    path: str
    role: str
    identity: str
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class InventoryRow:
    condition: str
    samples: int


@dataclass(frozen=True)
class LoadedManifest:
    files: list[CorpusFile] = field(default_factory=list)
    inventory: list[InventoryRow] = field(default_factory=list)


def load_manifest(manifest_path: Path, *, repo_root: Path) -> LoadedManifest:
    """Load and validate; refuse any file path inside the repository (exit 5)."""
    raw = json.loads(manifest_path.read_text())
    files: list[CorpusFile] = []
    counts = {condition: 0 for condition in CONDITION_ORDER}
    for entry in raw.get("files", []):
        role = entry.get("role")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        resolved = Path(str(entry["path"])).expanduser().resolve()
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError:
            pass
        else:
            raise ConfigurationError(f"corpus file inside repository: {resolved}")
        conditions = tuple(entry.get("conditions", []))
        for condition in conditions:
            if condition in counts:
                counts[condition] += 1
        files.append(
            CorpusFile(
                path=str(entry["path"]),
                role=role,
                identity=str(entry.get("identity", "")),
                conditions=conditions,
            )
        )
    inventory = [InventoryRow(condition=c, samples=counts[c]) for c in CONDITION_ORDER]
    return LoadedManifest(files=files, inventory=inventory)


def untested_conditions(loaded: LoadedManifest) -> list[str]:
    return [row.condition for row in loaded.inventory if row.samples == 0]

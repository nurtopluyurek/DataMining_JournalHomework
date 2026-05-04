from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_NAME = "CompSciencePub (1).sqlite"
DEFAULT_ARCHIVE_NAME = "CS_JournalAbstracts.zip"
DEFAULT_PDF_NAME = "Data_mining_journal_homework 2.pdf"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    raw_dir: Path
    artifacts_dir: Path
    cache_dir: Path
    figures_dir: Path
    notebook_path: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[2]
        data_dir = root / "data"
        raw_dir = data_dir / "raw"
        artifacts_dir = root / "artifacts"
        cache_dir = artifacts_dir / "cache"
        figures_dir = root / "figures"
        notebook_path = root / "final_project.ipynb"
        for path in (data_dir, raw_dir, artifacts_dir, cache_dir, figures_dir):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            data_dir=data_dir,
            raw_dir=raw_dir,
            artifacts_dir=artifacts_dir,
            cache_dir=cache_dir,
            figures_dir=figures_dir,
            notebook_path=notebook_path,
        )


def _downloads_dir() -> Path:
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Downloads"


def resolve_input_path(filename: str) -> Path:
    env_map = {
        DEFAULT_DB_NAME: "JOURNAL_REC_DB_PATH",
        DEFAULT_ARCHIVE_NAME: "JOURNAL_REC_ARCHIVE_PATH",
        DEFAULT_PDF_NAME: "JOURNAL_REC_PDF_PATH",
    }
    env_var = env_map.get(filename)
    if env_var and os.environ.get(env_var):
        path = Path(os.environ[env_var]).expanduser().resolve()
        if path.exists():
            return path

    paths = ProjectPaths.discover()
    candidates = [
        paths.raw_dir / filename,
        paths.data_dir / filename,
        _downloads_dir() / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate {filename!r} in {candidates}.")

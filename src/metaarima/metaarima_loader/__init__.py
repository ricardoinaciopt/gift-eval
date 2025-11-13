from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path
from typing import Any

import joblib

__all__ = ["load_metaarima_model", "MetaARIMAModel"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_package(repo_root: Path) -> None:
    """Register the `src` package so pickled models can import their code."""
    if "src" in sys.modules:
        return
    package_dir = repo_root / "src"
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        raise ImportError(
            f"Cannot locate 'src' package beside the loader at {package_dir}"
        )
    spec = importlib.util.spec_from_file_location(
        "src",
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import spec for 'src' package.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["src"] = module
    spec.loader.exec_module(module)


class MetaARIMAModel:
    """Lazy loader that proxies the MetaARIMA model stored as a gzipped joblib pickle."""

    def __init__(self, model_path: Path) -> None:
        self._path = model_path
        self._model: Any | None = None

    def load(self) -> Any:
        if self._model is None:
            repo_root = _repo_root()
            _ensure_src_package(repo_root)
            if not self._path.exists():
                raise FileNotFoundError(f"Model file not found: {self._path}")
            with gzip.open(self._path, "rb") as fh:
                self._model = joblib.load(fh)
        return self._model

    def __getattr__(self, name: str) -> Any:
        model = self.load()
        return getattr(model, name)


def load_metaarima_model(model_path: str | Path) -> Any:
    """Load and return the underlying MetaARIMA model instance."""
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return MetaARIMAModel(path.resolve()).load()

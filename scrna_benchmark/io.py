"""Small, atomic outputs and explicit run provenance."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import os
import tempfile


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False, suffix=".json") as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)


def write_h5ad(path, adata):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, suffix=".h5ad")
    os.close(fd)
    try:
        adata.write_h5ad(name, compression="gzip")
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def names_hash(names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def versions():
    packages = ("scvi-tools", "scanpy", "scib", "anndata", "torch", "jax", "jaxlib",
                "flax", "numpy", "pandas", "scipy", "snakemake")
    found = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found

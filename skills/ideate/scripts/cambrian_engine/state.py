"""File-based local state for a project.

State is written **outside** the plugin so reinstalls don't wipe it. The base
directory is ``~/.cambrian`` by default, overridable with the
``CAMBRIAN_HOME`` environment variable (used by the test suite to
isolate runs). Layout::

    <home>/<project>/
        meta.json                 # project metadata
        axes.json                 # snapshot of the resolved axes
        archive.json              # MAP-Elites: niche_id -> niche record
        candidates.json           # id -> candidate record (genealogy kept)
        embeddings.npz            # id -> embedding vector (binary; legacy .json still read)
        mech_embeddings.npz       # id -> mechanism/open-axis vector (advisory S4 store)
        tmp/                      # scratch dir for the skill's hand-off files
        memory/<domain>/
            comparisons.jsonl     # appended preference events
            pins.json             # pinned "stepping stone" ids

Preferences are namespaced per domain so switching domains keeps memories
separate.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigError

_log = logging.getLogger(__name__)

DEFAULT_HOME_ENV = "CAMBRIAN_HOME"
_DEFAULT_BASE = "~/.cambrian"

_PATH_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Best-effort cross-process lock (atomic mkdir works on every target OS) used to
# serialize read-modify-write on the small per-domain files (pins). Tuned for the
# realistic "two /ideate sessions on one project" race, not high contention.
_LOCK_TIMEOUT = 10.0   # seconds to wait for the lock before proceeding anyway
_LOCK_STALE = 60.0     # a lock dir older than this is assumed abandoned (crash)
_LOCK_POLL = 0.05
# Orphaned atomic-write temp files older than this are swept on State.ensure().
_STALE_TEMP_SECS = 3600.0


def _path_slug(name: str, fallback: str = "default") -> str:
    """Filesystem-safe slug for a project or domain id.

    A name that is already a valid slug round-trips unchanged, so existing state
    dirs (the common case: plain ASCII ids) are preserved exactly. When slugging is
    *lossy* — characters were replaced/stripped, or a non-ASCII-only id reduces to
    ``fallback`` — distinct names could otherwise collide on one slug and silently
    merge state (e.g. every non-ASCII id -> ``"default"``, or ``"proj A"`` and
    ``"proj-A"`` -> the same dir). In that case we append a short hash of the raw
    name so different ids can never share a directory.
    """
    raw = str(name)
    s = _PATH_SLUG_RE.sub("-", raw.strip()).strip("-_.")
    if s == raw:
        return s  # already a clean slug: unchanged, backward-compatible
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{s or fallback}-{suffix}"


def base_dir() -> Path:
    env = os.environ.get(DEFAULT_HOME_ENV)
    root = Path(env) if env else Path(_DEFAULT_BASE)
    return root.expanduser()


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so a rename is durable across a crash.

    ``os.fsync`` on the file only flushes its *contents*; the directory entry that
    ``os.replace`` creates is separate metadata and can be lost on power loss even
    though the data was synced. Fsyncing the parent closes that gap on POSIX. A
    no-op where a directory handle can't be fsynced (e.g. Windows has no
    ``O_DIRECTORY`` and raises on fsync of a dir fd).
    """
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        dfd = os.open(str(directory), os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


_REPLACE_TIMEOUT = 5.0     # seconds — a DEADLINE, not an attempt count
_REPLACE_BACKOFF = 0.05    # initial sleep; doubles per attempt, capped below
_REPLACE_BACKOFF_MAX = 0.25


def _replace_with_retry(tmp: str, path: Path) -> None:
    """``os.replace(tmp, path)`` with a bounded retry for the Windows sharing-violation case.

    On Windows, replacing a destination another process holds open WITHOUT ``FILE_SHARE_DELETE``
    raises ``PermissionError`` (ERROR_SHARING_VIOLATION) — e.g. a concurrent session reading the
    archive, or the AV/search indexer briefly opening the freshly-renamed file. Retrying to a
    ~5s DEADLINE keeps a momentary concurrent open from failing an otherwise-valid state write.
    A no-op on POSIX, where ``os.replace`` over an open file succeeds."""
    deadline = time.monotonic() + _REPLACE_TIMEOUT
    backoff = _REPLACE_BACKOFF
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(backoff, _REPLACE_BACKOFF_MAX))
            backoff *= 2


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            # Flush + fsync before the rename so a crash can't persist the rename ahead of the
            # data (which would leave a zero-length/stale file).
            fh.flush()
            os.fsync(fh.fileno())
        # Preserve the destination's existing permission bits across the inode-replacing rename.
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except OSError:
            pass
        _replace_with_retry(tmp, path)
        # Fsync the directory so the rename itself is durable, not just the data.
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _steal_stale_lock(lock: Path) -> None:
    """Atomically reclaim a presumed-abandoned lock dir.

    ``os.replace`` of a directory is atomic, so exactly one racer wins the rename
    and removes it; the losers get an ``OSError`` and simply re-attempt the
    ``mkdir``. This avoids the stat-then-``rmdir`` TOCTOU where two processes both
    observe the lock as stale, both delete it, and both proceed — one possibly
    deleting a *fresh* lock a third process just created.
    """
    sidelined = lock.with_name(f"{lock.name}.stale-{os.getpid()}")
    try:
        os.replace(lock, sidelined)
    except OSError:
        return  # another racer won the steal (or it vanished); just retry mkdir
    shutil.rmtree(sidelined, ignore_errors=True)


@contextlib.contextmanager
def _file_lock(target: Path, timeout: float = _LOCK_TIMEOUT):
    """Best-effort cross-process lock guarding a read-modify-write on ``target``.

    Uses an atomic ``mkdir`` of a sibling ``<name>.lock`` directory (atomic on
    POSIX and Windows). Steals a lock older than ``_LOCK_STALE`` (a crashed
    holder). If the lock can't be acquired within ``timeout`` we proceed anyway
    rather than deadlock — the protected section is then no worse than the
    historical unlocked behavior (and is logged at WARNING for diagnosis).
    """
    lock = target.parent / (target.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, timeout)
    acquired = False
    while True:
        try:
            lock.mkdir()
            acquired = True
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except OSError:
                age = 0.0
            if age > _LOCK_STALE:
                _steal_stale_lock(lock)
                continue
            if time.monotonic() >= deadline:
                _log.warning("lock not acquired within %.1fs; proceeding "
                             "unlocked on %s", timeout, target)
                break  # give up; proceed unlocked
            time.sleep(_LOCK_POLL)
        except OSError:
            # A transient FS error rather than "already held": on Windows a name
            # that another racer is mid-rmdir on can report a sharing/pending-delete
            # error instead of FileExistsError. Back off briefly and retry rather
            # than letting the read-modify-write crash.
            if time.monotonic() >= deadline:
                _log.warning("lock unavailable (transient FS errors); proceeding "
                             "unlocked on %s", target)
                break
            time.sleep(_LOCK_POLL)
    try:
        yield
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                lock.rmdir()


class State:
    """Handle to one project's on-disk state."""

    def __init__(self, project: str, home: Optional[Path] = None):
        self.project = project
        self.base = Path(home).expanduser() if home else base_dir()
        self.root = self.base / _path_slug(project)

    # -- paths -------------------------------------------------------------- #
    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def axes_path(self) -> Path:
        return self.root / "axes.json"

    @property
    def archive_path(self) -> Path:
        return self.root / "archive.json"

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.json"

    @property
    def embeddings_path(self) -> Path:
        """Binary npz vector store. The pre-npz pretty-JSON shape re-parsed and rewrote tens of MB
        of float reprs per ingest inside the project lock at large archives; values on disk are
        float64, so the round-trip is exact. ``_legacy_embeddings_json`` is read as a fallback so a
        project written by an older engine still loads."""
        return self.root / "embeddings.npz"

    @property
    def _legacy_embeddings_json(self) -> Path:
        return self.root / "embeddings.json"

    @property
    def mech_embeddings_path(self) -> Path:
        return self.root / "mech_embeddings.npz"

    @property
    def _legacy_mech_embeddings_json(self) -> Path:
        return self.root / "mech_embeddings.json"

    @property
    def open_nicher_path(self) -> Path:
        return self.root / "open_nicher.json"

    @property
    def tmp_dir(self) -> Path:
        """Per-project scratch dir for the skill's hand-off files (axes.json,
        candidates.json, event.json). Inside the state home so they never clutter
        the user's cwd and can't collide across concurrent sessions."""
        return self.root / "tmp"

    def memory_dir(self, domain: str) -> Path:
        return self.root / "memory" / _path_slug(domain)

    def comparisons_path(self, domain: str) -> Path:
        return self.memory_dir(domain) / "comparisons.jsonl"

    def pins_path(self, domain: str) -> Path:
        return self.memory_dir(domain) / "pins.json"

    def discards_path(self, domain: str) -> Path:
        return self.memory_dir(domain) / "discards.json"

    # -- lifecycle ---------------------------------------------------------- #
    def exists(self) -> bool:
        return self.meta_path.exists()

    def ensure(self) -> "State":
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._sweep_stale_temps()
        return self

    def _sweep_stale_temps(self) -> None:
        """Remove orphaned atomic-write temp files. A crash between ``mkstemp``
        and ``os.replace`` leaves a ``.tmp-*`` file behind; sweep old ones so they
        don't accumulate. Best-effort: races and errors are ignored."""
        cutoff = time.time() - _STALE_TEMP_SECS
        try:
            entries = list(self.root.glob(".tmp-*"))
        except OSError:
            return
        for p in entries:
            with contextlib.suppress(OSError):
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()

    def paths(self) -> Dict[str, str]:
        return {
            "root": str(self.root),
            "meta": str(self.meta_path),
            "axes": str(self.axes_path),
            "archive": str(self.archive_path),
            "candidates": str(self.candidates_path),
            "embeddings": str(self.embeddings_path),
            "mech_embeddings": str(self.mech_embeddings_path),
            "open_nicher": str(self.open_nicher_path),
            "tmp": str(self.tmp_dir),
        }

    def reset_geometry(self) -> None:
        """Delete this project's geometric state — the MAP-Elites archive, candidate
        records, embeddings (surface AND the parallel mechanism store), and the frozen
        open-axis nicher — so the project can be re-initialized under a NEW axes
        geometry without mixing stale niche keys with new ones. Preference memory
        (pins/discards/comparisons, namespaced per domain) is deliberately left intact.
        Best-effort: a missing file is not an error."""
        for path in (self.archive_path, self.candidates_path,
                     self.embeddings_path, self.mech_embeddings_path,
                     # legacy pre-npz vector files: an older project's leftovers must not shadow a
                     # fresh geometry via the tolerant fallback read.
                     self._legacy_embeddings_json, self._legacy_mech_embeddings_json,
                     self.open_nicher_path):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    def project_lock(self, timeout: float = _LOCK_TIMEOUT):
        """Cross-process lock serializing a whole-project read-modify-write cycle.

        ``ingest`` (and an axes-changing ``init``) read the archive/candidates/
        embeddings/meta, mutate them, and write them back; without serialization two
        concurrent cycles on one project would clobber each other (a lost generation).
        Same best-effort ``mkdir`` lock as the pin/discard guard — on timeout it
        proceeds unlocked rather than deadlock. The lock dir sits beside the state
        files as ``<root>/.project.lock``."""
        return _file_lock(self.root / ".project", timeout=timeout)

    # -- generic json helpers ---------------------------------------------- #
    def read_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            # Empty file (e.g. an interrupted write) — treat as absent rather
            # than crash the command; the next write repopulates it.
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # A non-empty but corrupt state file is surfaced as a clean,
            # operator-facing error instead of a raw traceback.
            raise ConfigError(
                f"state file {path} is corrupt ({exc}); remove it (or restore a "
                f"backup) and retry"
            ) from exc

    def write_json(self, path: Path, obj: Any, *, compact: bool = False) -> None:
        """``compact=True`` drops the indent/whitespace for big machine-only stores — values and
        key order (sort_keys) are identical, only bytes shrink."""
        _atomic_write(path, json.dumps(
            obj, ensure_ascii=False, sort_keys=True,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        ))

    # -- vector stores (npz) ------------------------------------------------- #
    def _read_vector_store(self, path: Path, legacy_json: Path) -> Dict[str, List[float]]:
        """id -> vector store, npz-backed. ``.tolist()`` of the float64 rows returns the identical
        Python floats ``json.loads`` produced from the legacy file, so consumers (and the
        write/read roundtrip) are unchanged. Falls back to the legacy pretty-JSON file so a project
        written by an older engine still reads; the next write lands the npz and removes the
        legacy file."""
        if path.exists():
            import io

            import numpy as np
            try:
                with np.load(io.BytesIO(path.read_bytes())) as z:
                    # keys carry a "v:" prefix — see _write_vector_store.
                    return {k[2:]: z[k].tolist() for k in z.files}
            except Exception as exc:  # noqa: BLE001 — corrupt zip/npy: mirror read_json's posture
                raise ConfigError(
                    f"state file {path} is corrupt ({exc}); remove it (or restore a "
                    f"backup) and retry"
                ) from exc
        return self.read_json(legacy_json, {}) or {}

    def _write_vector_store(self, path: Path, legacy_json: Path,
                            embeddings: Dict[str, List[float]]) -> None:
        import io

        import numpy as np
        buf = io.BytesIO()
        # Candidate ids are caller-supplied strings; the "v:" prefix keeps any id (e.g. a literal
        # "file") from colliding with np.savez's own parameter names while round-tripping verbatim.
        np.savez(buf, **{f"v:{k}": np.asarray(v, dtype=np.float64)
                         for k, v in embeddings.items()})
        _atomic_write_bytes(path, buf.getvalue())
        with contextlib.suppress(OSError):
            legacy_json.unlink(missing_ok=True)

    # -- typed accessors ---------------------------------------------------- #
    def read_meta(self) -> Dict[str, Any]:
        return self.read_json(self.meta_path, {}) or {}

    def write_meta(self, meta: Dict[str, Any]) -> None:
        self.write_json(self.meta_path, meta)

    def read_axes(self) -> Optional[Dict[str, Any]]:
        return self.read_json(self.axes_path, None)

    def write_axes(self, axes: Dict[str, Any]) -> None:
        self.write_json(self.axes_path, axes)

    def read_archive(self) -> Dict[str, Any]:
        return self.read_json(self.archive_path, {}) or {}

    def write_archive(self, archive: Dict[str, Any]) -> None:
        self.write_json(self.archive_path, archive)

    def read_candidates(self) -> Dict[str, Any]:
        return self.read_json(self.candidates_path, {}) or {}

    def write_candidates(self, candidates: Dict[str, Any]) -> None:
        # compact: the largest JSON store (full idea texts + genealogy), machine-only.
        self.write_json(self.candidates_path, candidates, compact=True)

    def read_embeddings(self) -> Dict[str, List[float]]:
        return self._read_vector_store(self.embeddings_path, self._legacy_embeddings_json)

    def write_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        self._write_vector_store(self.embeddings_path, self._legacy_embeddings_json, embeddings)

    def read_mech_embeddings(self) -> Dict[str, List[float]]:
        return self._read_vector_store(self.mech_embeddings_path,
                                       self._legacy_mech_embeddings_json)

    def write_mech_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        self._write_vector_store(self.mech_embeddings_path,
                                 self._legacy_mech_embeddings_json, embeddings)

    def read_open_nicher(self) -> Optional[Dict[str, Any]]:
        """Persisted open-axis nicher: cold-start accumulation or frozen centroids."""
        return self.read_json(self.open_nicher_path, None)

    def write_open_nicher(self, data: Dict[str, Any]) -> None:
        # compact: the pre-freeze accumulation buffer is up to freeze_factor*k raw vectors.
        self.write_json(self.open_nicher_path, data, compact=True)

    # -- memory (namespaced by domain) ------------------------------------- #
    def append_comparison(self, domain: str, event: Dict[str, Any]) -> None:
        path = self.comparisons_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        # Single O_APPEND write (not buffered text I/O): under POSIX O_APPEND each
        # write lands atomically at end-of-file, so concurrent appenders can't
        # interleave a short line. read_comparisons also tolerates a torn line.
        data = line.encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

    def read_comparisons(self, domain: str) -> List[Dict[str, Any]]:
        path = self.comparisons_path(domain)
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip a truncated/corrupt line (e.g. an interrupted append) so a
                # single bad record can't poison every future read of this domain.
                continue
        return out

    def read_pins(self, domain: str) -> List[str]:
        return self.read_json(self.pins_path(domain), []) or []

    def write_pins(self, domain: str, pins: List[str]) -> None:
        self.write_json(self.pins_path(domain), pins)

    def add_pin(self, domain: str, candidate_id: str) -> List[str]:
        # Lock the read-modify-write so two concurrent invocations can't each read
        # the same list and clobber the other's pin (pins are "never dropped").
        path = self.pins_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(path):
            pins = self.read_pins(domain)
            if candidate_id not in pins:
                pins.append(candidate_id)
                self.write_pins(domain, pins)
            # Pins and discards are mutually exclusive (latest action wins): pinning
            # an id the user previously discarded un-discards it.
            self._remove_discard(domain, candidate_id)
            return pins

    def read_discards(self, domain: str) -> List[str]:
        return self.read_json(self.discards_path(domain), []) or []

    def write_discards(self, domain: str, discards: List[str]) -> None:
        self.write_json(self.discards_path(domain), discards)

    def add_discard(self, domain: str, candidate_id: str) -> List[str]:
        # Locked read-modify-write, mirroring add_pin. A discard is the negative of
        # a pin; the two are mutually exclusive (latest action wins), so discarding
        # an id also drops it from pins.
        path = self.discards_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(path):
            discards = self.read_discards(domain)
            if candidate_id not in discards:
                discards.append(candidate_id)
                self.write_discards(domain, discards)
            self._remove_pin(domain, candidate_id)
            return discards

    def _remove_pin(self, domain: str, candidate_id: str) -> None:
        pins = self.read_pins(domain)
        if candidate_id in pins:
            self.write_pins(domain, [p for p in pins if p != candidate_id])

    def _remove_discard(self, domain: str, candidate_id: str) -> None:
        discards = self.read_discards(domain)
        if candidate_id in discards:
            self.write_discards(domain, [d for d in discards if d != candidate_id])

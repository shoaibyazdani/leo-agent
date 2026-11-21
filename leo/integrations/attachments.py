"""Local, validated chat attachments.

Uploaded bytes stay on disk under the configured workspace. Only metadata and a
bounded text preview enter the model prompt; untrusted file content is never
executed and is always labelled as data.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import sqlite3
import stat
from pathlib import Path
from typing import Any

from .. import db
from ..config import Settings
from ..contracts import new_id, utcnow

MAX_BYTES = 10 * 1024 * 1024
TEXT_MIME = {
    "text/plain", "text/markdown", "text/csv", "application/json",
    "application/xml", "text/xml",
}
ALLOWED_MIME = TEXT_MIME | {
    "image/png", "image/jpeg", "image/webp", "image/gif",
    "application/pdf",
}
IMAGE_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGE_PAYLOAD = 5 * 1024 * 1024
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
)
# Only these fields are safe to hand to a browser. The host path, the text
# preview and any future internal field must stay on the server.
_PUBLIC_FIELDS = (
    "id", "conversation_id", "name", "mime", "size", "sha256", "kind", "created_at",
)


def _sniff_mime(raw: bytes) -> str | None:
    """Inspect the whole (already size-bounded) upload, not just its prefix.

    Magic bytes beyond a 512-byte header used to slip through as "text" and
    were then returned verbatim in a text preview.
    """
    offset = raw.find(b"RIFF")
    while offset >= 0:
        if raw[offset + 8:offset + 12] == b"WEBP":
            return "image/webp"
        offset = raw.find(b"RIFF", offset + 1)
    for signature, mime in _MAGIC:
        if signature in raw:
            return mime
    return None


class AttachmentError(ValueError):
    """Raised for an invalid or unsupported attachment."""


class AttachmentStore:
    @staticmethod
    def public(record: dict[str, Any]) -> dict[str, Any]:
        """Never expose the absolute host path or any internal field."""
        return {
            key: record[key]
            for key in _PUBLIC_FIELDS
            if key in record and record[key] is not None
        }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.resolve(settings.paths.workspaces).resolve()

    def _safe_name(self, name: Any) -> str:
        text = str(name or "attachment").split("/")[-1].split("\\")[-1]
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
        return cleaned[:80] or "attachment"

    def _target_dir(self, conversation_id: str | None) -> Path:
        # conversation_id is untrusted (it arrives from the API). Interpolating
        # it verbatim let an absolute path replace the base, or `..` segments
        # escape it, and the old code even mkdir()ed *before* checking. Reduce
        # it to a single safe path component and verify containment first.
        component = self._safe_name(conversation_id) if conversation_id else "unassigned"
        base = self.root / "attachments" / component
        resolved = base.resolve()
        if not resolved.is_relative_to(self.root):
            raise AttachmentError("attachment path outside allowed workspace")
        # Only create directories after the containment check has passed.
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    async def save(
        self, name: Any, mime: Any, data_base64: Any, conversation_id: str | None = None
    ) -> dict[str, Any]:
        mime_value = str(mime or "application/octet-stream").split(";")[0].strip().lower()
        if mime_value not in ALLOWED_MIME:
            raise AttachmentError(f"unsupported attachment type: {mime_value}")
        if not isinstance(data_base64, str) or not data_base64:
            raise AttachmentError("attachment data must be base64 encoded")
        if len(data_base64) > int(MAX_BYTES * 4 / 3) + 16:
            raise AttachmentError("encoded attachment is too large")
        try:
            raw = base64.b64decode(data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AttachmentError("invalid base64 attachment data") from exc
        if not raw:
            raise AttachmentError("attachment is empty")
        if len(raw) > MAX_BYTES:
            raise AttachmentError(f"attachment exceeds {MAX_BYTES // (1024 * 1024)} MB limit")

        sniffed = _sniff_mime(raw)
        if sniffed is not None and sniffed != mime_value:
            raise AttachmentError("file content does not match its declared type")
        if sniffed is None and mime_value not in TEXT_MIME:
            raise AttachmentError("unrecognised or unsupported file content")
        # Scan every byte, not just the first 4 KiB: a NUL later in the file
        # must not smuggle binary data into a "text" preview.
        if mime_value in TEXT_MIME and b"\x00" in raw:
            raise AttachmentError("text attachment contains binary data")

        digest = hashlib.sha256(raw).hexdigest()
        safe_name = self._safe_name(name)
        target_dir = self._target_dir(conversation_id)
        # Content-addressed filename inside the already-validated directory.
        path = target_dir / f"{digest[:16]}_{safe_name}"
        # A deterministic path can be pre-planted as a symlink by anyone who can
        # write the workspace. O_NOFOLLOW makes that fail closed instead of
        # overwriting the symlink's target.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        created_new = not path.exists()
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise AttachmentError("cannot create attachment file") from exc
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
        except OSError as exc:
            raise AttachmentError("cannot write attachment file") from exc
        try:
            path.chmod(0o600)
        except OSError:
            pass

        preview = ""
        if mime_value in TEXT_MIME:
            preview = raw.decode("utf-8", errors="replace")[:16000]
        record = {
            "id": new_id("att"),
            "conversation_id": conversation_id,
            "name": safe_name,
            "mime": mime_value,
            "size": len(raw),
            "sha256": digest,
            "path": str(path),
            "text_preview": preview,
            "kind": "image" if mime_value in IMAGE_MIME else "file",
            "created_at": utcnow(),
        }
        sql = (
            "INSERT INTO attachments(id,conversation_id,name,mime,size,sha256,path,text_preview,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)"
        )
        try:
            # A primary-key collision is astronomically unlikely with uuid4,
            # but never leave an orphaned file if metadata cannot be stored.
            for _ in range(3):
                record["id"] = new_id("att")
                try:
                    await db.execute(sql, (
                        record["id"], record["conversation_id"], record["name"], record["mime"],
                        record["size"], record["sha256"], record["path"], preview,
                        record["created_at"],
                    ))
                    break
                except sqlite3.IntegrityError:  # duplicate id -> regenerate
                    continue
            else:
                raise AttachmentError("could not allocate a unique attachment id")
        except Exception:
            if created_new:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return record

    async def get(self, attachment_id: str) -> dict[str, Any] | None:
        row = await db.fetchone("SELECT * FROM attachments WHERE id=?", (attachment_id,))
        if not row:
            return None
        record = dict(row)
        mime = str(record.get("mime", "")).lower()
        record["kind"] = "image" if mime in IMAGE_MIME else "file"
        return record

    def resolve_content(self, record: dict[str, Any]) -> Path | None:
        """Validate a stored row again at request time and return its safe path.

        Upload-time checks are not enough: the file may have been replaced by a
        symlink, turned into a FIFO, grown, or the row may have been tampered
        with. Every one of those cases returns ``None`` so the caller answers
        404 rather than serving arbitrary workspace content.
        """
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        if record.get("mime") not in ALLOWED_MIME:
            return None
        try:
            # A symlink swapped in after upload is refused even if it points
            # back inside the workspace.
            if stat.S_ISLNK(Path(raw_path).lstat().st_mode):
                return None
            resolved = Path(raw_path).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return None
        if not resolved.is_relative_to(self.root):
            return None
        current = self.root
        for part in resolved.relative_to(self.root).parts:
            current = current / part
            try:
                if stat.S_ISLNK(current.lstat().st_mode):
                    return None
            except OSError:
                return None
        try:
            info = resolved.stat()
        except OSError:
            return None
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            return None
        expected = record.get("size")
        if (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and expected >= 0
            and info.st_size != expected
        ):
            return None
        return resolved

    def image_block(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Return an Anthropic image content block for a stored image.

        Only regular files inside the store's workspace root are read; a record
        whose path points elsewhere (for example a tampered database row) is
        refused. The declared media type must be an allowlisted image type and
        match the file's sniffed content. Callers resolve records through
        :meth:`get`; raw client-supplied path/mime dicts never reach here.
        """
        mime = record.get("mime")
        if not isinstance(mime, str) or mime not in IMAGE_MIME:
            return None
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        try:
            resolved = Path(raw_path).resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        if not resolved.is_relative_to(self.root):
            return None
        try:
            info = resolved.stat()
        except OSError:
            return None
        # Check the size before reading so a tampered row pointing at a large
        # file cannot force an unbounded read into memory. Refuse non-regular
        # files (FIFOs/devices) as well.
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_IMAGE_PAYLOAD:
            return None
        try:
            with resolved.open("rb") as handle:
                raw = handle.read(MAX_IMAGE_PAYLOAD + 1)
        except OSError:
            return None
        if len(raw) > MAX_IMAGE_PAYLOAD or _sniff_mime(raw) != mime:
            return None
        data = base64.b64encode(raw).decode("ascii")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": data,
            },
        }

"""Hardening regressions for the local attachment store.

Covers MIME/content validation, size caps, base64 errors, filename and
conversation-id traversal, workspace containment, image-block construction,
bounded text previews, public-field filtering, duplicate ids and cleanup.
"""

from __future__ import annotations

import base64
import hashlib
import sqlite3
from pathlib import Path

import pytest

from leo.config import PathConfig, Settings
from leo.integrations import attachments as att
from leo.integrations.attachments import AttachmentError, AttachmentStore

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def make_store(tmp_path: Path) -> AttachmentStore:
    settings = Settings(paths=PathConfig(workspaces=tmp_path / "workspaces"))
    return AttachmentStore(settings)


# --------------------------------------------------------------------------- #
# conversation_id / filename traversal and workspace containment
# --------------------------------------------------------------------------- #

async def test_conversation_id_cannot_escape_workspace(tmp_path: Path) -> None:
    """Absolute/relative/odd conversation ids never create dirs outside root."""
    store = make_store(tmp_path)
    hostile_ids = [
        str(tmp_path / "outside_marker"),  # absolute path replaces the base
        "../../escaped",                    # parent traversal
        "..",                               # resolves to workspace root itself
        "/etc",                             # absolute system path
        12345,                              # non-string from a direct caller
    ]
    for cid in hostile_ids:
        record = await store.save("pixel.png", "image/png", b64(PNG), cid)
        path = Path(record["path"])
        assert path.is_relative_to(store.root), path
        assert path.is_file()

    # The old code mkdir()ed the hostile path *before* the containment check.
    for leaked in ("outside_marker", "escaped", "etc"):
        assert not (tmp_path / leaked).exists(), f"created {leaked} outside workspace"


async def test_filename_traversal_and_weird_names(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for name in ["../../etc/passwd", "..\\..\\windows\\system32\\x.exe", "....", "", None]:
        record = await store.save(name, "text/plain", b64(b"hello"))
        path = Path(record["path"])
        assert path.is_relative_to(store.root)
        assert path.is_file()
        assert "/" not in record["name"] and "\\" not in record["name"]
        assert record["name"] not in {".", "..", ""}


async def test_target_dir_normal_case_is_contained(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    d = store._target_dir("conv_abc")
    assert d.is_relative_to(store.root)
    assert d.is_dir()


# --------------------------------------------------------------------------- #
# base64 decoding, MIME sniffing, size caps
# --------------------------------------------------------------------------- #

async def test_invalid_inputs_are_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(AttachmentError):
        await store.save("x.png", "image/png", "not-base64!!")
    with pytest.raises(AttachmentError):
        await store.save("x.png", "image/png", b64(b"not a png"))
    with pytest.raises(AttachmentError):
        await store.save("x.exe", "application/x-msdownload", b64(b"MZ"))
    with pytest.raises(AttachmentError):
        await store.save("x.txt", "text/plain", b64(b""))
    with pytest.raises(AttachmentError):
        await store.save("x.txt", "text/plain", 12345)  # non-string data
    with pytest.raises(AttachmentError):
        await store.save("x.txt", "text/plain", b64(PNG))  # image bytes as text


async def test_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store(tmp_path)
    monkeypatch.setattr(att, "MAX_BYTES", 1024)
    with pytest.raises(AttachmentError):
        await store.save("big.txt", "text/plain", b64(b"x" * 2000))


async def test_content_sniff_mismatch_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    # PDF magic declared as PNG.
    with pytest.raises(AttachmentError):
        await store.save("fake.png", "image/png", b64(b"%PDF-1.7 fake"))
    # WEBP is detected too, so it cannot masquerade as text/plain.
    webp = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8 " + b"\x00" * 16
    with pytest.raises(AttachmentError):
        await store.save("a.txt", "text/plain", b64(webp))


# --------------------------------------------------------------------------- #
# text previews and public metadata
# --------------------------------------------------------------------------- #

async def test_text_binary_null_beyond_4k_is_rejected(tmp_path: Path) -> None:
    """A NUL after the first 4 KiB must not smuggle binary into a preview."""
    store = make_store(tmp_path)
    payload = (b"a" * 5000) + b"\x00" + b"b"
    with pytest.raises(AttachmentError):
        await store.save("notes.txt", "text/plain", b64(payload))


async def test_text_preview_bounded_and_not_public(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    text = "hello world\n" * 3000  # ~36 KiB
    record = await store.save("notes.txt", "text/plain", b64(text.encode()))
    assert "text_preview" in record
    assert len(record["text_preview"]) == 16000
    public = AttachmentStore.public(record)
    assert "text_preview" not in public and "path" not in public
    assert public["kind"] == "file" and public["mime"] == "text/plain"


async def test_public_uses_allowlist(tmp_path: Path) -> None:
    record = {
        "id": "att_x", "name": "a.txt", "mime": "text/plain", "size": 1,
        "path": "/secret/host/path", "text_preview": "secret text",
        "internal_token": "do-not-leak", "kind": "file", "created_at": "now",
    }
    public = AttachmentStore.public(record)
    assert "path" not in public and "text_preview" not in public
    assert "internal_token" not in public
    assert public["id"] == "att_x" and public["kind"] == "file"


# --------------------------------------------------------------------------- #
# image block construction
# --------------------------------------------------------------------------- #

async def test_image_block_validates_media_type_and_content(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = await store.save("pixel.png", "image/png", b64(PNG))
    block = store.image_block(record)
    assert block and block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"]

    # Only exact, allowlisted image types are accepted (no svg/bmp, no case games).
    assert store.image_block({"mime": "image/svg+xml", "path": record["path"]}) is None
    assert store.image_block({"mime": "IMAGE/PNG", "path": record["path"]}) is None
    assert store.image_block({"mime": 123, "path": record["path"]}) is None

    # Bytes must actually match the declared image type.
    fake = store.root / "attachments" / "unassigned" / "fake.png"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"not really a png" * 10)
    assert store.image_block({"mime": "image/png", "path": str(fake)}) is None


async def test_image_block_checks_size_before_full_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An oversized file is refused without reading the whole thing into memory."""
    store = make_store(tmp_path)
    big = store.root / "attachments" / "unassigned" / "big.png"
    big.parent.mkdir(parents=True, exist_ok=True)
    with big.open("wb") as handle:
        handle.write(PNG[:8])
        handle.truncate(att.MAX_IMAGE_PAYLOAD + 4096)

    def forbidden_read(self: Path, *args: object, **kwargs: object) -> bytes:
        raise AssertionError("full read_bytes() was attempted")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    assert store.image_block({"mime": "image/png", "path": str(big)}) is None


async def test_image_block_refuses_outside_and_symlink(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    outside = tmp_path / "secret.png"
    outside.write_bytes(PNG)
    assert store.image_block({"mime": "image/png", "path": str(outside)}) is None

    inside = store.root / "attachments" / "link.png"
    inside.parent.mkdir(parents=True, exist_ok=True)
    try:
        inside.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert store.image_block({"mime": "image/png", "path": str(inside)}) is None


# --------------------------------------------------------------------------- #
# symlink-safe writes, duplicate ids, cleanup on failure
# --------------------------------------------------------------------------- #

async def test_symlink_plant_cannot_redirect_write(tmp_path: Path) -> None:
    """A pre-planted symlink at the deterministic path is not followed."""
    store = make_store(tmp_path)
    digest = hashlib.sha256(PNG).hexdigest()
    folder = store.root / "attachments" / "unassigned"
    folder.mkdir(parents=True, exist_ok=True)
    victim = store.root / "attachments" / "victim.txt"
    victim.write_bytes(b"ORIGINAL")
    link = folder / f"{digest[:16]}_pixel.png"
    try:
        link.symlink_to(victim)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(AttachmentError):
        await store.save("pixel.png", "image/png", b64(PNG))
    assert victim.read_bytes() == b"ORIGINAL"


async def test_duplicate_id_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store(tmp_path)
    real_execute = att.db.execute

    async def fake_execute(sql: str, params: tuple = ()) -> object:
        if sql.startswith("INSERT INTO attachments") and params[0] == "att_dup":
            raise sqlite3.IntegrityError("UNIQUE constraint failed: attachments.id")
        return await real_execute(sql, params)

    monkeypatch.setattr(att.db, "execute", fake_execute)
    ids = iter(["att_dup", "att_dup", "att_unique"])
    monkeypatch.setattr(att, "new_id", lambda prefix="": next(ids))

    record = await store.save("a.txt", "text/plain", b64(b"hello"))
    assert record["id"] == "att_unique"


async def test_failed_metadata_insert_removes_orphan_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store(tmp_path)

    async def boom(sql: str, params: tuple = ()) -> object:
        raise sqlite3.IntegrityError("boom")

    monkeypatch.setattr(att.db, "execute", boom)
    with pytest.raises(AttachmentError):
        await store.save("a.txt", "text/plain", b64(b"hello"))

    leftovers = [p for p in (store.root / "attachments").rglob("*") if p.is_file()]
    assert leftovers == []


async def test_sniff_covers_magic_beyond_first_512_bytes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    hidden_pdf = b"a" * 700 + b"%PDF-1.7 smuggled past the old 512-byte sniff"
    with pytest.raises(AttachmentError, match="content does not match"):
        await store.save("notes.txt", "text/plain", b64(hidden_pdf))


async def test_content_endpoint_revalidates_file_and_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from leo import db
    from leo.interfaces import api

    store = make_store(tmp_path)
    record = await store.save("pixel.png", "image/png", b64(PNG), "conv_live")

    class AppStub:
        attachments = store

    monkeypatch.setattr(api, "get_app", lambda: AppStub())
    response = await api.attachment_content(record["id"])
    assert response.status_code == 200

    path = Path(record["path"])
    outside = tmp_path / "secret.png"
    outside.write_bytes(PNG)

    # Symlink swap after upload must not serve the outside file.
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(HTTPException) as link_error:
        await api.attachment_content(record["id"])
    assert link_error.value.status_code == 404

    # Non-regular file at the stored path is refused too.
    path.unlink()
    path.mkdir()
    with pytest.raises(HTTPException) as dir_error:
        await api.attachment_content(record["id"])
    assert dir_error.value.status_code == 404
    path.rmdir()

    # A tampered row pointing outside the workspace is refused.
    path.write_bytes(PNG)
    await db.execute(
        "UPDATE attachments SET path=? WHERE id=?", (str(outside), record["id"])
    )
    with pytest.raises(HTTPException) as outside_error:
        await api.attachment_content(record["id"])
    assert outside_error.value.status_code == 404

    # A tampered size cannot force a large/unbounded read.
    await db.execute(
        "UPDATE attachments SET path=?, size=? WHERE id=?",
        (str(path), 10 * 1024 * 1024 + 1, record["id"]),
    )
    with pytest.raises(HTTPException) as size_error:
        await api.attachment_content(record["id"])
    assert size_error.value.status_code == 404

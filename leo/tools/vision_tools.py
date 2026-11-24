"""``vision_analyze``: let the agent look at an image through the vision slot.

The tool exists because the turn path only sees images the operator attaches. An
agent that wrote a screenshot, or that is asked about a file on disk, needs the
same reader on demand — and it must use the SAME provider the operator configured
for images, not the chat model. Risk is LOW: a description writes nothing.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contracts import RiskLevel, ToolSpec
from ..integrations.attachments import IMAGE_MIME, MAX_IMAGE_PAYLOAD, _sniff_mime

_SUFFIX_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _services(ctx: Any) -> Mapping[str, Any]:
    value = ctx.get("services") if isinstance(ctx, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _require_vision(ctx: Any) -> Any:
    vision = _services(ctx).get("vision")
    describe = getattr(vision, "describe", None)
    if not callable(describe):
        raise ValueError("vision is not configured in this process")
    if getattr(vision, "slot", lambda: None)() is None:
        raise ValueError(
            "no vision provider: set models.vision.provider or enable a provider that declares 'vision'"
        )
    return vision


def _block_from_path(raw_path: str) -> tuple[list[dict[str, Any]], str]:
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ValueError(f"not a file: {raw_path}")
    mime = _SUFFIX_MIME.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"not an image suffix: {path.suffix or '(none)'}")
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_PAYLOAD:
        raise ValueError(f"image larger than {MAX_IMAGE_PAYLOAD} bytes")
    # The suffix is a claim; the bytes decide. A renamed text file must not be
    # handed to the vision model as if it were a picture.
    if _sniff_mime(data) != mime:
        raise ValueError(f"file content is not {mime}")
    return (
        [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        ],
        path.name,
    )


async def _load_blocks(args: Mapping[str, Any], ctx: Any) -> tuple[list[dict[str, Any]], str]:
    attachment_id = args.get("attachment_id")
    if isinstance(attachment_id, str) and attachment_id.strip():
        store = _services(ctx).get("attachments")
        record = None
        if store is not None and callable(getattr(store, "get", None)):
            record = await store.get(attachment_id.strip())
        if not record:
            raise ValueError(f"no stored attachment: {attachment_id}")
        if store is None or not callable(getattr(store, "image_block", None)):
            raise ValueError("the attachment store is not available in this process")
        block = store.image_block(record)
        if not block:
            raise ValueError("that attachment is not a readable image")
        return [block], str(record.get("name") or attachment_id)
    raw_path = args.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        return _block_from_path(raw_path.strip())
    raise ValueError("give either path or attachment_id")


async def vision_analyze(args: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Describe an image (file path or stored attachment) with the vision model."""
    vision = _require_vision(ctx)
    blocks, label = await _load_blocks(args, ctx)
    question = args.get("question")
    description = await vision.describe(
        blocks, question=str(question) if isinstance(question, str) and question.strip() else None
    )
    if not description:
        return {"ok": False, "image": label, "error": "the vision model returned no description"}
    return {"ok": True, "image": label, "description": description}


def register(registry: Any, services: Any = None) -> list[str]:
    """Register ``vision_analyze``; concrete services arrive per call in ``ctx``."""
    spec = ToolSpec(
        name="vision_analyze",
        description=(
            "Read an image and return what it shows, using the dedicated vision model. "
            "Pass 'path' for a local file or 'attachment_id' for an image sent in chat; "
            "'question' narrows the read (e.g. 'what error is on screen?')."
        ),
        category="vision",
        risk=RiskLevel.LOW,
        permissions=["vision:read"],
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "attachment_id": {"type": "string", "minLength": 1},
                "question": {"type": "string", "minLength": 1},
            },
            "anyOf": [{"required": ["path"]}, {"required": ["attachment_id"]}],
        },
    )
    registry.register(spec, vision_analyze)
    return [spec.name]

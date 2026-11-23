"""The smart-mode reviewer's model response is parsed conservatively."""

from __future__ import annotations

from types import SimpleNamespace

from leo.config import Settings
from leo.core.app import LeoApp


class _Router:
    def __init__(self, result) -> None:
        self.result = result
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return SimpleNamespace(content=self.result)


def _app(result) -> SimpleNamespace:
    return SimpleNamespace(settings=Settings(), router=_Router(result))


async def _review(result) -> tuple[str, str]:
    app = _app(result)
    return await LeoApp._review_approval(
        app,
        command="rsvg-convert -z 2 -o girl.png girl.svg",
        resource="/root/workspace",
        risk_reason="unclassified command (rsvg-convert)",
        action="tool:shell",
    )


async def test_reviewer_parses_the_one_line_verdicts() -> None:
    assert (await _review("ALLOW"))[0] == "allow"
    assert (await _review("**ALLOW**"))[0] == "allow"
    assert (await _review("DENY: writes outside the workspace"))[0] == "deny"
    assert (await _review("ASK"))[0] == "ask"


async def test_reviewer_falls_back_to_ask_on_prose_empty_or_failure() -> None:
    for raw in ("I think it is probably fine", "", "```json\n{}\n```"):
        assert (await _review(raw))[0] == "ask", raw
    assert (await _review(RuntimeError("reviewer down")))[0] == "ask"


async def test_reviewer_sees_the_command_as_bounded_data() -> None:
    app = _app("ALLOW")
    await LeoApp._review_approval(
        app,
        command="rsvg-convert -z 2 -o girl.png girl.svg",
        resource="/root/workspace",
        risk_reason="unclassified command (rsvg-convert)",
        action="tool:shell",
    )
    request = app.router.requests[0]
    body = request.messages[-1].content
    assert "rsvg-convert" in body
    assert "/root/workspace" in body
    assert request.max_tokens <= 200
    assert request.tools is None or request.tools == []

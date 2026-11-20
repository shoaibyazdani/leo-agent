"""Guided onboarding: one-time codes, local QR, honest states, safe validation.

Every value below is synthetic. No network, browser or provider call is made;
the QR is rendered in process and all sessions live in the isolated test DB.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from leo import db
from leo.config import ChannelsConfig
from leo.events import EventBus
from leo.integrations import onboarding as ob
from leo.integrations.admission import ChannelAdmission
from leo.integrations.channels import ALL_ADAPTERS, default_adapters
from leo.integrations.enrollment import (
    EnrollmentError,
    channel_details,
    enable_channel,
)
from leo.integrations.gateway import Gateway
from leo.integrations.secrets import default_store
from leo.interfaces import api as api_module

FAKE_TOKEN = "fake-whatsapp-cloud-token-DO-NOT-LEAK-9f8a7b"
FAKE_PHONE_ID = "123456789012345"
FAKE_SECRET = "fake-meta-app-secret-DO-NOT-LEAK-7d6e5f"
BASE_URL = "http://127.0.0.1:8000"
ONBOARD_HEADERS = {"X-Leo-Onboarding": "1"}


@pytest.fixture(autouse=True)
def fresh_limits():
    """Limiter state is process-global; every test starts and ends uncapped."""
    limiters = (ob._issue_limiter, ob._redeem_client_limiter, ob._redeem_code_limiter)
    for limiter in limiters:
        limiter._hits.clear()
    yield
    for limiter in limiters:
        limiter._hits.clear()


@pytest.fixture
def gateway() -> Gateway:
    default_store.clear()
    return Gateway(
        default_adapters(),
        events=EventBus(persist=False),
        admission=ChannelAdmission(ChannelsConfig()),
    )


@pytest.fixture
def offline_app(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Only the gateway is needed; avoid unrelated app/agent startup work."""
    monkeypatch.delenv("LEO_API_TOKEN", raising=False)
    default_store.clear()
    gateway = Gateway(
        default_adapters(),
        events=EventBus(persist=False),
        admission=ChannelAdmission(ChannelsConfig()),
    )
    app = SimpleNamespace(gateway=gateway)
    monkeypatch.setattr(api_module, "_app", app)
    return app


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_module.api), base_url=BASE_URL
    )


def _code_from(result: dict) -> str:
    fragment = urlsplit(result["qr"]["payload"]).fragment
    return parse_qs(fragment)["code"][0]


async def _record(code: str) -> dict:
    found = await ob._load_session(ob._code_hash(code))
    assert found is not None
    return found


# --- Token lifecycle --------------------------------------------------------


async def test_start_status_redeem_lifecycle() -> None:
    started = await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:1")
    assert started["channel"] == "whatsapp"
    assert started["state"] == ob.STATE_AWAITING_SCAN
    assert [item["state"] for item in started["transitions"]] == [
        ob.STATE_STARTED, ob.STATE_AWAITING_SCAN
    ]
    assert 0 < started["expires_in"] <= ob.SETUP_TOKEN_TTL_SECONDS
    assert started["status_url"].endswith(started["session_id"])

    session_id = started["session_id"]
    status = await ob.session_status(session_id)
    assert status is not None
    assert status["state"] == ob.STATE_AWAITING_SCAN
    assert status["linked"] is False and status["ready"] is False

    code = _code_from(started)
    confirmed = await ob.redeem_setup_code(code, client="unit:1")
    assert confirmed["state"] == ob.STATE_PAIRED
    assert confirmed["confirmed_at"]
    assert confirmed["linked"] is False and confirmed["ready"] is False

    after = await ob.session_status(session_id)
    assert after is not None and after["state"] == ob.STATE_PAIRED
    assert [item["state"] for item in after["transitions"]][-1] == ob.STATE_PAIRED

    # Replay never re-issues or re-opens the code.
    with pytest.raises(ob.OnboardingError) as replay:
        await ob.redeem_setup_code(code, client="unit:1")
    assert replay.value.status_code == 403
    assert replay.value.code in ("replayed", "invalid_code")


async def test_raw_code_is_never_persisted_or_returned_in_status() -> None:
    started = await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:2")
    code = _code_from(started)
    rows = await db.fetchall("SELECT key,value FROM kv WHERE key LIKE ?", ("onboarding.session.%",))
    assert len(rows) == 1
    assert code not in rows[0]["value"]
    assert ob._code_hash(code) in rows[0]["key"]
    status = await ob.session_status(started["session_id"])
    assert code not in json.dumps(status)


async def test_expired_code_fails_and_status_says_so() -> None:
    started = await ob.start_onboarding(
        "whatsapp", base_url=BASE_URL, client="unit:3", ttl_seconds=1
    )
    code = _code_from(started)
    record = await _record(code)
    record["expires_epoch"] = time.time() - 1
    await ob._save_session(record)

    with pytest.raises(ob.OnboardingError) as expired:
        await ob.redeem_setup_code(code, client="unit:3")
    assert expired.value.code == "expired"
    assert expired.value.status_code == 403

    status = await ob.session_status(started["session_id"])
    assert status is not None
    assert status["state"] == ob.STATE_FAILED
    assert "expired" in status["message"].lower()
    assert status["linked"] is False


async def test_invalid_code_is_rejected_before_lookup() -> None:
    with pytest.raises(ob.OnboardingError) as invalid:
        await ob.redeem_setup_code("not-a-real-code", client="unit:4")
    assert invalid.value.status_code == 403
    with pytest.raises(ob.OnboardingError) as unknown:
        await ob.redeem_setup_code("A" * 43, client="unit:4")
    assert unknown.value.code == "invalid_code"


async def test_rate_limits_and_pending_cap() -> None:
    for _ in range(ob.ISSUE_LIMIT_PER_MINUTE):
        await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:limit")
    with pytest.raises(ob.OnboardingError) as limited:
        await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:limit")
    assert limited.value.code == "rate_limited"
    assert limited.value.status_code == 429

    ob._issue_limiter._hits.clear()
    await db.execute("DELETE FROM kv WHERE key LIKE ?", ("onboarding.session.%",))
    old_cap = ob.MAX_PENDING_SESSIONS
    ob.MAX_PENDING_SESSIONS = 1
    try:
        await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:cap:1")
        with pytest.raises(ob.OnboardingError) as capped:
            await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:cap:2")
        assert capped.value.code == "too_many_sessions"
    finally:
        ob.MAX_PENDING_SESSIONS = old_cap


async def test_per_code_redeem_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    started = await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:limit")
    code = _code_from(started)
    monkeypatch.setattr(ob._redeem_code_limiter, "limit", 1)
    await ob.redeem_setup_code(code, client="unit:limit")
    with pytest.raises(ob.OnboardingError) as limited:
        await ob.redeem_setup_code(code, client="unit:limit")
    assert limited.value.code == "rate_limited"


async def test_concurrent_redemptions_confirm_exactly_once() -> None:
    started = await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:race")
    code = _code_from(started)

    async def attempt() -> str:
        try:
            await ob.redeem_setup_code(code, client="unit:race")
            return "paired"
        except ob.OnboardingError as exc:
            return exc.code or "error"

    # Client limiter allows 10/min, so both attempts reach the DB gate.
    outcomes = await asyncio.gather(attempt(), attempt())
    assert outcomes.count("paired") == 1
    assert set(outcomes) - {"paired"} <= {"replayed", "invalid_code"}


async def test_ttl_is_clamped() -> None:
    started = await ob.start_onboarding(
        "whatsapp", base_url=BASE_URL, client="unit:ttl", ttl_seconds=10_000
    )
    assert started["expires_in"] <= ob.SETUP_TOKEN_MAX_SECONDS
    ob._issue_limiter._hits.clear()
    shortest = await ob.start_onboarding(
        "whatsapp", base_url=BASE_URL, client="unit:ttl", ttl_seconds=-1
    )
    assert 0 <= shortest["expires_in"] <= 1


# --- QR and origin safety ---------------------------------------------------


def test_qr_matrix_is_structurally_valid() -> None:
    payload = f"{BASE_URL}/onboarding/pair#code={'A' * 43}"
    matrix = ob.qr_matrix(payload)
    size = len(matrix)
    assert size >= 21 and size % 2 == 1
    # Three finder patterns: 7x7 rings with a 3x3 centre.
    for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):
        assert matrix[cy][cx] is True
        assert matrix[cy - 3][cx - 3] is True
        assert matrix[cy - 2][cx - 2] is False
        assert matrix[cy - 1][cx - 1] is True
        assert matrix[cy - 3][cx] is True
        assert matrix[cy][cx - 3] is True
    assert sum(sum(row) for row in matrix) > 50

    svg, data_uri = ob.qr_svg_data_uri(payload)
    assert data_uri.startswith("data:image/svg+xml;base64,")
    assert svg == base64.b64decode(data_uri.split(",", 1)[1]).decode("utf-8")
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
    assert '<path fill="#000000"' in svg
    assert ob.qr_svg_data_uri(payload)[0] == svg
    assert ob.qr_svg_data_uri(payload + "x")[0] != svg
    with pytest.raises(ValueError, match="too long"):
        ob.qr_matrix("x" * 5000)


def _gf_mul(a: int, b: int) -> int:
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        b >>= 1
        high = a & 0x80
        a = (a << 1) & 0xFF
        if high:
            a ^= 0x1D
    return result


def _mask_bit(mask: int, x: int, y: int) -> bool:
    """Independent implementation of the eight QR mask formulas."""
    return (
        (x + y) % 2 == 0,
        y % 2 == 0,
        x % 3 == 0,
        (x + y) % 3 == 0,
        (y // 2 + x // 3) % 2 == 0,
        x * y % 2 + x * y % 3 == 0,
        (x * y % 2 + x * y % 3) % 2 == 0,
        ((x + y) % 2 + x * y % 3) % 2 == 0,
    )[mask]


def _decode_qr(matrix: list[list[bool]]) -> str:
    """Decode byte-mode ECC-M data from a matrix (independent of the encoder)."""
    size = len(matrix)
    format_bits = 0
    positions = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8), (7, 8),
        (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
    ]
    for index, (x, y) in enumerate(positions):
        format_bits |= int(matrix[y][x]) << index
    format_bits ^= 0x5412
    assert (format_bits >> 13) & 3 == 0  # medium ECC
    mask = (format_bits >> 10) & 7

    fixed = [[False] * size for _ in range(size)]
    for i in range(size):
        fixed[i][6] = fixed[6][i] = True
    for i in range(6):
        fixed[i][8] = True
    for i in (7, 8):
        fixed[i][8] = fixed[8][i] = True
    for i in range(9, 15):
        fixed[8][14 - i] = True
    for i in range(8):
        fixed[8][size - 1 - i] = True
    for i in range(8, 15):
        fixed[size - 15 + i][8] = True
    fixed[size - 8][8] = True
    for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if 0 <= cx + dx < size and 0 <= cy + dy < size:
                    fixed[cy + dy][cx + dx] = True
    version = (size - 17) // 4
    if version >= 2:
        count = version // 7 + 2
        step = ((version * 4 + count * 2 + 1) // (count * 2 - 2)) * 2
        marks = [6] + [size - 7 - i * step for i in reversed(range(count - 1))]
        for x in marks:
            for y in marks:
                if (x, y) in ((6, 6), (6, size - 7), (size - 7, 6)):
                    continue
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        fixed[y + dy][x + dx] = True
    if version >= 7:
        for i in range(18):
            a, b = size - 11 + i % 3, i // 3
            fixed[a][b] = fixed[b][a] = True

    bits: list[int] = []
    column, upward = size - 1, True
    while column > 0:
        if column == 6:
            column -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for y in rows:
            for x in (column, column - 1):
                if fixed[y][x]:
                    continue
                dark = matrix[y][x] != _mask_bit(mask, x, y)
                bits.append(int(dark))
        column -= 2
        upward = not upward

    data_total = ob._data_capacity(version)
    ecc_len = ob._QR_ECC_CODEWORDS[version - 1][ob._QR_ECL_INDEX]
    blocks = ob._QR_BLOCKS[version - 1][ob._QR_ECL_INDEX]
    assert len(bits) // 8 == data_total + ecc_len * blocks
    words = [
        sum(bits[i * 8 + j] << (7 - j) for j in range(8))
        for i in range(len(bits) // 8)
    ]
    short_len, long_blocks = divmod(data_total, blocks)
    short_count = blocks - long_blocks
    lengths = [short_len + (0 if i < short_count else 1) for i in range(blocks)]
    groups: list[list[int]] = [[] for _ in range(blocks)]
    cursor = 0
    for i in range(short_len + 1):
        for index in range(blocks):
            if i < lengths[index]:
                groups[index].append(words[cursor])
                cursor += 1
    ecc = [
        [words[data_total + i * blocks + b] for i in range(ecc_len)]
        for b in range(blocks)
    ]
    # Every Reed-Solomon syndrome must vanish; this caught a real block-split
    # bug that a purely structural QR test could not see.
    for index in range(blocks):
        codeword = groups[index] + ecc[index]
        for root in range(ecc_len):
            alpha = 1
            for _ in range(root):
                alpha = _gf_mul(alpha, 2)
            value = 0
            for byte in codeword:
                value = _gf_mul(value, alpha) ^ byte
            assert value == 0

    stream = [byte for group in groups for byte in group]
    raw = "".join(f"{byte:08b}" for byte in stream)
    assert int(raw[:4], 2) == 4  # byte mode
    length = int(raw[4:12], 2)
    return bytes(int(raw[12 + i * 8:20 + i * 8], 2) for i in range(length)).decode()


def test_setup_qr_decodes_back_to_its_exact_url() -> None:
    # Covers several QR versions, including a multi-block one matching the
    # real setup URL length, so block splitting/parity regressions are caught.
    for length in (1, 40, 90, 110):
        payload = f"{BASE_URL}/onboarding/pair#code={'A' * 43}"[:length].ljust(
            length, "B"
        )
        assert _decode_qr(ob.qr_matrix(payload)) == payload


async def test_qr_payload_only_carries_opaque_code_and_loopback_url() -> None:
    started = await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:qr")
    code = _code_from(started)
    payload = started["qr"]["payload"]
    assert payload.startswith(f"{BASE_URL}/onboarding/pair#code=")
    assert code in payload
    assert started["qr"]["format"] == "svg-data-uri"
    for forbidden in (FAKE_TOKEN, FAKE_SECRET, "WHATSAPP_CLOUD_TOKEN", "password", "/" + code):
        assert forbidden not in payload
    # The setup URL is a loopback URL derived from the server, not from input.
    hostile = await ob.start_onboarding(
        "whatsapp", base_url="https://evil.example/steal?x=1@", client="unit:qr:hostile"
    )
    assert hostile["qr"]["payload"].startswith("http://localhost:5173/onboarding/pair")


def test_origin_checks_are_exact_and_loopback_only() -> None:
    assert ob.safe_base_url("http://localhost:9000/path?x=1") == "http://localhost:9000"
    assert ob.safe_base_url("https://user:pass@evil.example/path") == "http://localhost:5173"
    assert ob.safe_base_url("http://[::1]:8000/path") == "http://[::1]:8000"
    assert ob.origin_allowed(None, base_url=BASE_URL)
    assert ob.origin_allowed("http://localhost:5173", base_url=BASE_URL)
    assert ob.origin_allowed(BASE_URL, base_url=BASE_URL)
    assert not ob.origin_allowed("https://evil.example", base_url=BASE_URL)
    assert not ob.origin_allowed("http://127.0.0.1:8000.evil.example", base_url=BASE_URL)
    assert not ob.origin_allowed("null", base_url=BASE_URL)
    assert not ob.origin_allowed("*", base_url=BASE_URL)


async def test_origin_mismatch_does_not_consume_code() -> None:
    started = await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:origin")
    code = _code_from(started)
    with pytest.raises(ob.OnboardingError) as mismatch:
        await ob.redeem_setup_code(code, client="unit:origin", origin="https://evil.example")
    assert mismatch.value.code == "origin_mismatch"
    status = await ob.session_status(started["session_id"])
    assert status is not None and status["state"] == ob.STATE_AWAITING_SCAN
    confirmed = await ob.redeem_setup_code(code, client="unit:origin", origin=BASE_URL)
    assert confirmed["state"] == ob.STATE_PAIRED


async def test_code_is_bound_to_its_issuing_client() -> None:
    started = await ob.start_onboarding("whatsapp", base_url=BASE_URL, client="unit:owner")
    code = _code_from(started)
    with pytest.raises(ob.OnboardingError) as mismatch:
        await ob.redeem_setup_code(code, client="unit:someone-else")
    assert mismatch.value.code == "client_mismatch"
    assert (await ob.session_status(started["session_id"]))["state"] == ob.STATE_AWAITING_SCAN
    confirmed = await ob.redeem_setup_code(code, client="unit:owner")
    assert confirmed["state"] == ob.STATE_PAIRED


async def test_unsupported_channel_cannot_start_qr_setup() -> None:
    with pytest.raises(ob.OnboardingError) as exc:
        await ob.start_onboarding("telegram", base_url=BASE_URL, client="unit:channel")
    assert exc.value.status_code == 404
    assert exc.value.code == "unsupported_channel"


# --- Honest device-link state ----------------------------------------------


def test_device_link_is_blocked_without_a_real_bridge() -> None:
    status = ob.device_link_status("whatsapp")
    assert status["status"] == "blocked"
    assert status["linked"] is False
    assert status["backend"] is None
    assert "no WhatsApp Web" in status["reason"]
    assert "not a WhatsApp device-link QR" in status["reason"]

    class FakeBridge:
        pass

    original = getattr(ALL_ADAPTERS["whatsapp"], "device_link_backend", None)
    ALL_ADAPTERS["whatsapp"].device_link_backend = FakeBridge
    try:
        available = ob.device_link_status("whatsapp")
        assert available["status"] == "available"
        assert available["linked"] is False
        assert available["backend"] == "FakeBridge"
    finally:
        if original is None:
            del ALL_ADAPTERS["whatsapp"].device_link_backend
        else:
            ALL_ADAPTERS["whatsapp"].device_link_backend = original


# --- Guided field schema and validation ------------------------------------


def test_field_guidance_is_plain_language_and_value_free() -> None:
    guidance = ob.field_guidance(
        "WHATSAPP_PHONE_NUMBER_ID", kind="text", label="Phone number ID"
    )
    assert "NOT your WhatsApp phone number" in guidance["hint"]
    assert guidance["example"].isdigit()
    assert guidance["validation"]["pattern"]
    assert "not your WhatsApp phone number" in guidance["validation"]["message"]
    assert guidance["sensitive"] is False
    assert ob.field_guidance("WHATSAPP_CLOUD_TOKEN", kind="secret")["sensitive"] is True
    assert "Paste the Meta access token" in ob.field_guidance(
        "WHATSAPP_CLOUD_TOKEN", kind="secret"
    )["placeholder"]


def test_validate_values_reports_field_level_errors() -> None:
    assert ob.validate_values("whatsapp", {
        "WHATSAPP_PHONE_NUMBER_ID": FAKE_PHONE_ID,
        "WHATSAPP_API_VERSION": "v20.0",
        "mode": "allowlist",
        "allowlist": "+15551234567",
    }) == []

    errors = ob.validate_values("whatsapp", {
        "WHATSAPP_PHONE_NUMBER_ID": "+15551234567",  # a phone number, not an ID
        "WHATSAPP_API_VERSION": "20.0",
        "mode": "everyone",
        "allowlist": "not-a-phone-number",
        "MADE_UP_FIELD": "x",
        "WHATSAPP_CLOUD_TOKEN": FAKE_TOKEN + "\nattack",
    })
    names = {error["field"] for error in errors}
    assert names == {
        "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_API_VERSION", "mode",
        "allowlist", "MADE_UP_FIELD", "WHATSAPP_CLOUD_TOKEN",
    }
    assert all(error["message"] for error in errors)
    assert FAKE_TOKEN not in json.dumps(errors)

    assert ob.validate_values("email", {"allowlist": "not-an-email-is-allowed"}) == []
    assert ob.validate_values("email", {"SMTP_PORT": "70000"})
    assert ob.validate_values("telegram", {"TELEGRAM_WEBHOOK_SECRET": FAKE_SECRET}) == []


async def test_channel_details_exposes_guided_schema(gateway: Gateway) -> None:
    details = await channel_details("whatsapp", gateway=gateway)
    assert details["docs_url"].endswith("/cloud-api")
    assert details["guide"]["steps"][0]["title"].startswith("1.")
    assert details["guide"]["notes"][0] == ob.BLOCKED_DEVICE_LINK_REASON
    assert details["device_link"]["status"] == "blocked"
    by_name = {field["name"]: field for field in details["fields"]}
    phone = by_name["WHATSAPP_PHONE_NUMBER_ID"]
    assert phone["label"] == "Phone number ID"
    assert phone["required"] is True
    assert phone["example"] == FAKE_PHONE_ID
    assert "NOT your WhatsApp phone number" in phone["hint"]
    assert phone["validation"]["message"]
    version = by_name["WHATSAPP_API_VERSION"]
    assert version["required"] is False and version["default"] == "v20.0"
    assert by_name["WHATSAPP_CLOUD_TOKEN"]["sensitive"] is True
    assert FAKE_TOKEN not in json.dumps(details)


async def test_enable_rejects_bad_fields_without_writing(gateway: Gateway) -> None:
    values = {
        "WHATSAPP_CLOUD_TOKEN": FAKE_TOKEN,
        "WHATSAPP_PHONE_NUMBER_ID": "+15551234567",
        "WHATSAPP_APP_SECRET": FAKE_SECRET,
    }
    with pytest.raises(EnrollmentError) as bad:
        await enable_channel("whatsapp", values, allowlist=["+15551234567"], gateway=gateway)
    assert bad.value.field_errors
    assert bad.value.field_errors[0]["field"] == "WHATSAPP_PHONE_NUMBER_ID"
    assert FAKE_TOKEN not in str(bad.value)
    assert default_store.get("WHATSAPP_CLOUD_TOKEN") is None

    values["WHATSAPP_PHONE_NUMBER_ID"] = FAKE_PHONE_ID
    details = await enable_channel("whatsapp", values, allowlist=["+15551234567"], gateway=gateway)
    assert details["status"] == "configured"
    assert FAKE_TOKEN not in json.dumps(details)
    assert FAKE_SECRET not in json.dumps(details)


# --- HTTP surface -----------------------------------------------------------


async def test_api_guide_qr_status_and_redeem(offline_app: Any) -> None:
    async with _client() as client:
        guide = await client.get("/onboarding/whatsapp/guide")
        assert guide.status_code == 200, guide.text
        body = guide.json()
        assert body["docs_url"].endswith("/cloud-api")
        assert body["callback_url"] == f"{BASE_URL}/webhooks/whatsapp"
        assert body["qr_setup"]["available"] is True
        assert body["qr_setup"]["device_link"]["linked"] is False
        assert "QR" in body["guide"]["notes"][0]
        assert body["fields"][0]["hint"]

        started = await client.post(
            "/onboarding/whatsapp/start", headers=ONBOARD_HEADERS, json={}
        )
        assert started.status_code == 200, started.text
        result = started.json()
        assert result["qr"]["data_uri"].startswith("data:image/svg+xml;base64,")
        assert "linked" in result and result["linked"] is False

        status = await client.get(result["status_url"])
        assert status.status_code == 200, status.text
        assert status.json()["state"] == ob.STATE_AWAITING_SCAN

        redeemed = await client.post(
            "/onboarding/redeem",
            headers=ONBOARD_HEADERS,
            json={"code": _code_from(result)},
        )
        assert redeemed.status_code == 200, redeemed.text
        assert redeemed.json()["state"] == ob.STATE_PAIRED
        assert redeemed.json()["linked"] is False
        assert redeemed.json()["ready"] is False

        pair_page = await client.get("/onboarding/pair")
        assert pair_page.status_code == 200
        assert "does not link a device" in pair_page.text
        assert "X-Leo-Onboarding" in pair_page.text


async def test_api_csrf_origin_and_loopback_guards(offline_app: Any) -> None:
    async with _client() as client:
        missing_header = await client.post("/onboarding/whatsapp/start", json={})
        assert missing_header.status_code == 403
        assert "X-Leo-Onboarding" in missing_header.json()["detail"]

        bad_origin = await client.post(
            "/onboarding/whatsapp/start",
            headers={**ONBOARD_HEADERS, "Origin": "https://evil.example"},
            json={},
        )
        assert bad_origin.status_code == 403

        bad_host = await client.post(
            "/onboarding/whatsapp/start",
            headers={**ONBOARD_HEADERS, "Host": "evil.example"},
            json={},
        )
        assert bad_host.status_code == 403

        good = await client.post(
            "/onboarding/whatsapp/start",
            headers={**ONBOARD_HEADERS, "Origin": "http://localhost:5173"},
            json={"ttl_seconds": 30},
        )
        assert good.status_code == 200, good.text

        unknown = await client.get("/onboarding/sessions/not-a-session")
        assert unknown.status_code == 404


async def test_api_validation_and_enable_errors(offline_app: Any) -> None:
    bad_values = {
        "WHATSAPP_CLOUD_TOKEN": FAKE_TOKEN,
        "WHATSAPP_PHONE_NUMBER_ID": "12345",
        "WHATSAPP_APP_SECRET": FAKE_SECRET,
    }
    async with _client() as client:
        validate = await client.post(
            "/integrations/whatsapp/validate",
            json={"values": bad_values, "allowlist": ["not-a-phone"]},
        )
        assert validate.status_code == 200, validate.text
        body = validate.json()
        assert body["valid"] is False and body["offline"] is True
        assert {error["field"] for error in body["errors"]} >= {
            "WHATSAPP_PHONE_NUMBER_ID", "allowlist"
        }
        assert FAKE_TOKEN not in validate.text and FAKE_SECRET not in validate.text

        rejected = await client.post(
            "/integrations/whatsapp/enable",
            json={**bad_values, "allowlist": ["not-a-phone"]},
        )
        assert rejected.status_code == 400
        detail = rejected.json()["detail"]
        assert detail["errors"]
        assert FAKE_TOKEN not in rejected.text and FAKE_SECRET not in rejected.text

        unknown = await client.post(
            "/onboarding/not-a-channel/start", headers=ONBOARD_HEADERS, json={}
        )
        assert unknown.status_code == 404


async def test_api_never_leaks_credentials_through_qr_after_enrollment(
    offline_app: Any,
) -> None:
    async with _client() as client:
        enabled = await client.post(
            "/integrations/whatsapp/enable",
            json={
                "values": {
                    "WHATSAPP_CLOUD_TOKEN": FAKE_TOKEN,
                    "WHATSAPP_PHONE_NUMBER_ID": FAKE_PHONE_ID,
                    "WHATSAPP_APP_SECRET": FAKE_SECRET,
                },
                "allowlist": ["+15551234567"],
            },
        )
        assert enabled.status_code == 200, enabled.text
        flow = await client.post(
            "/onboarding/whatsapp/start", headers=ONBOARD_HEADERS, json={}
        )
        assert flow.status_code == 200
        payload = json.dumps(flow.json())
        assert FAKE_TOKEN not in payload
        assert FAKE_SECRET not in payload
        assert not any(name in payload for name in ("cloud_token", "app_secret"))


async def test_status_get_is_read_only_for_an_expired_session(
    offline_app: Any,
) -> None:
    """A browser polling status must not transition durable session state."""
    started = await ob.start_onboarding(
        "whatsapp", base_url=BASE_URL, client="unit:readonly"
    )
    code = _code_from(started)
    record = await _record(code)
    record["expires_epoch"] = time.time() - 1
    await ob._save_session(record)
    before = json.dumps(await _record(code), sort_keys=True)

    async with _client() as client:
        response = await client.get(started["status_url"])
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["state"] == ob.STATE_AWAITING_SCAN  # stored state unchanged
        assert body["expires_in"] == 0
        assert "expired" in body["message"].lower()

    assert json.dumps(await _record(code), sort_keys=True) == before

    # Housekeeping still happens in the explicit mutation endpoint.
    with pytest.raises(ob.OnboardingError) as expired:
        await ob.redeem_setup_code(code, client="unit:readonly")
    assert expired.value.code == "expired"
    assert (await _record(code))["state"] == ob.STATE_FAILED

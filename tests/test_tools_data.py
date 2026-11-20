"""Executor-level coverage for the offline data-transformation toolset.

Every tool is registered on a real ToolRegistry and dispatched through a real
ToolExecutorImpl (permissions, validation, audit and events included). Outputs
are asserted exactly; malformed input is asserted as an auditable tool error.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leo import db
from leo.config import Settings
from leo.contracts import RiskLevel, ToolCall
from leo.events import EventBus
from leo.security.permissions import PermissionManagerImpl
from leo.tools import ToolExecutorImpl, ToolRegistry
from leo.tools import data_tools

NAMES = ["json_query", "csv_parse", "text_transform", "base64_codec"]


@pytest.fixture
async def env(tmp_path, monkeypatch):
    db.set_db_path(tmp_path / "test.db")
    await db.initialize()
    settings = Settings()
    settings.security.allowed_roots = [tmp_path]
    events = EventBus()
    permissions = PermissionManagerImpl(settings, events)
    registry = ToolRegistry()
    names = data_tools.register(registry, services=None)
    executor = ToolExecutorImpl(registry, permissions, events, settings)

    # The integration agent injects `services` into the executor context; the
    # data toolset needs no service, but we inject at that same boundary so the
    # tests exercise the intended wiring shape.
    original_context = executor._context

    def with_services(call, task_key, dry_run):
        ctx = original_context(call, task_key, dry_run)
        ctx["services"] = {}
        return ctx

    monkeypatch.setattr(executor, "_context", with_services)
    return SimpleNamespace(
        tmp_path=tmp_path, settings=settings, events=events, permissions=permissions,
        registry=registry, executor=executor, names=names,
    )


async def _run(env, name, arguments=None, **kwargs):
    call = ToolCall(name=name, arguments=arguments or {}, actor="data-tester", **kwargs)
    return await env.executor.execute(call)


def _assert_error(result, fragment):
    assert result.success is False
    assert result.output is None
    assert result.error and fragment in result.error, result.error
    assert result.decision in ("invalid_arguments", "error", "denied")


# --------------------------------------------------------------------------- specs


async def test_register_typed_specs(env):
    assert env.names == NAMES
    assert [spec.name for spec in env.registry.list_specs()] == NAMES

    expected = {
        "json_query": ("data", RiskLevel.LOW, ["data:read"]),
        "csv_parse": ("data", RiskLevel.LOW, ["data:read"]),
        "text_transform": ("data", RiskLevel.LOW, ["data:transform"]),
        "base64_codec": ("data", RiskLevel.LOW, ["data:transform"]),
    }
    for name, (category, risk, permissions) in expected.items():
        spec = env.registry.get(name).spec
        assert spec.category == category
        assert spec.risk is risk
        assert spec.permissions == permissions
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False
        assert spec.input_schema["required"]
        assert spec.resource_field == ""

    assert env.registry.validate("json_query", {"data": {}, "path": "a"}) == []
    for name in NAMES:
        assert env.registry.validate(name, {})  # every tool has required fields
        assert env.registry.validate(name, {"unexpected": 1})  # additionalProperties=false
    assert data_tools.register(ToolRegistry(), services={}) == NAMES


# --------------------------------------------------------------------------- json_query


async def test_json_query_exact_outputs(env):
    document = {
        "user": {"name": "Ada", "tags": ["math", "code"]},
        "items": [{"id": 7}, {"id": 8}],
    }
    result = await _run(env, "json_query", {
        "data": json.dumps(document), "path": "user.tags.1",
    })
    assert result.success and result.decision == "allowed"
    assert result.output == {"found": True, "path": "user.tags.1", "value": "code"}

    result = await _run(env, "json_query", {"data": document, "path": "items.0.id"})
    assert result.output == {"found": True, "path": "items.0.id", "value": 7}

    result = await _run(env, "json_query", {"data": document, "path": ""})
    assert result.output == {"found": True, "path": "", "value": document}

    result = await _run(env, "json_query", {
        "data": document, "path": "user.missing", "default": "N/A",
    })
    assert result.output == {"found": False, "path": "user.missing", "value": "N/A"}

    result = await _run(env, "json_query", {"data": document, "path": "user.missing"})
    assert result.output == {"found": False, "path": "user.missing", "value": None}


async def test_json_query_malformed_input_errors(env):
    cases = [
        ({"data": "{not json", "path": "x"}, "not valid JSON"),
        ({"data": "   ", "path": "x"}, "not valid JSON"),
        ({"data": {"rows": [1, 2]}, "path": "rows.x"}, "not a valid list index"),
        ({"data": {"rows": [1, 2]}, "path": "rows.9"}, None),  # absent index, not malformed
        ({"data": {"a": 1}, "path": "a.b"}, "cannot traverse a int"),
        ({"data": {"a": 1}, "path": "a..b"}, "is empty"),
        ({"data": {"a": 1}, "path": ".".join(["k"] * 65)}, "segments"),
        ({"data": 123, "path": "x"}, "invalid arguments"),
        ({"data": "{}"}, "missing required field"),
    ]
    for arguments, fragment in cases:
        result = await _run(env, "json_query", arguments)
        if fragment is None:
            assert result.success is True
            assert result.output == {"found": False, "path": "rows.9", "value": None}
        else:
            _assert_error(result, fragment)


# --------------------------------------------------------------------------- csv_parse


async def test_csv_parse_exact_outputs(env):
    result = await _run(env, "csv_parse", {"text": "name,age\nAda,36\nGrace,85\n"})
    assert result.success and result.decision == "allowed"
    assert result.output == {
        "header": ["name", "age"],
        "rows": [
            {"name": "Ada", "age": "36"},
            {"name": "Grace", "age": "85"},
        ],
        "row_count": 2,
    }

    result = await _run(env, "csv_parse", {
        "text": "a|1\nb|2\n", "delimiter": "|", "header": False,
    })
    assert result.output == {
        "header": None, "rows": [["a", "1"], ["b", "2"]], "row_count": 2,
    }

    result = await _run(env, "csv_parse", {"text": 'city,"quoted, value"\nRome,"a,b"\n'})
    assert result.output["rows"] == [
        {"city": "Rome", "quoted, value": "a,b"}
    ]


async def test_csv_parse_malformed_input_errors(env):
    cases = [
        ({"text": 'a,b\n"unterminated,2\n'}, "malformed CSV"),
        ({"text": "a,b\n1\n"}, "row 2 has 1 field(s) but header has 2"),
        ({"text": "a,a\n1,2\n"}, "duplicate column names"),
        ({"text": ",b\n1,2\n"}, "empty column name"),
        ({"text": " \n"}, "CSV input is empty"),
        ({"text": "a,b\n", "delimiter": ",,"}, "invalid arguments"),
        ({"text": "a,b\n", "header": "yes"}, "invalid arguments"),
        ({"text": "a,b\n", "surprise": True}, "invalid arguments"),
    ]
    for arguments, fragment in cases:
        result = await _run(env, "csv_parse", arguments)
        _assert_error(result, fragment)


# --------------------------------------------------------------------------- text_transform


async def test_text_transform_exact_outputs(env):
    result = await _run(env, "text_transform", {
        "text": "Hello, World!", "operation": "upper",
    })
    assert result.success and result.decision == "allowed"
    assert result.output == {
        "operation": "upper", "text": "HELLO, WORLD!", "length": 13, "replacements": 0,
    }

    result = await _run(env, "text_transform", {
        "text": "hello world", "operation": "title",
    })
    assert result.output == {
        "operation": "title", "text": "Hello World", "length": 11, "replacements": 0,
    }

    result = await _run(env, "text_transform", {
        "text": "one fish two fish", "operation": "replace",
        "old": "fish", "new": "cat",
    })
    assert result.output == {
        "operation": "replace", "text": "one cat two cat", "length": 15, "replacements": 2,
    }

    result = await _run(env, "text_transform", {
        "text": "a1 b22 c333", "operation": "regex_replace",
        "pattern": "[0-9]+", "replacement": "#",
    })
    assert result.output == {
        "operation": "regex_replace", "text": "a# b# c#", "length": 8, "replacements": 3,
    }

    result = await _run(env, "text_transform", {
        "text": "2024-01-15", "operation": "regex_replace",
        "pattern": "(\\d{4})-(\\d{2})", "replacement": "\\2/\\1",
    })
    assert result.output == {
        "operation": "regex_replace", "text": "01/2024-15", "length": 10, "replacements": 1,
    }

    result = await _run(env, "text_transform", {
        "text": "a a a", "operation": "regex_replace",
        "pattern": "a", "replacement": "X", "max_replacements": 1,
    })
    assert result.output["text"] == "X a a"
    assert result.output["replacements"] == 1


async def test_text_transform_rejects_catastrophic_and_malformed_input(env):
    unsafe_patterns = [
        ("(a+)+$", "nested quantifiers"),
        ("(a|aa)+", "nested quantifiers"),
        ("(a|b)+", "nested quantifiers"),
        ("a+a+a+", "quantifiers"),
        ("\\1", "backreferences"),
        ("(?=a)b", "lookarounds"),
        ("(?i)abc", "special groups"),
        ("a{100000}", "quantifier bound"),
        ("a{2,1}", "quantifier bound"),
        ("[abc", "unterminated character class"),
        ("abc)", "unbalanced parenthesis"),
    ]
    for pattern, fragment in unsafe_patterns:
        result = await _run(env, "text_transform", {
            "text": "aaa", "operation": "regex_replace",
            "pattern": pattern, "replacement": "x",
        })
        _assert_error(result, fragment)

    cases = [
        ({"text": "x", "operation": "replace"}, "old"),
        ({"text": "x", "operation": "replace", "old": "", "new": "y"}, "must not be empty"),
        ({"text": "x", "operation": "regex_replace"}, "pattern"),
        ({"text": "x", "operation": "regex_replace", "pattern": "x"}, "replacement"),
        ({"text": "x", "operation": "regex_replace", "pattern": "x",
          "replacement": "y", "max_replacements": -1}, "invalid arguments"),
        ({"text": "x", "operation": "shout"}, "invalid arguments"),
        ({"operation": "upper"}, "invalid arguments"),
    ]
    for arguments, fragment in cases:
        result = await _run(env, "text_transform", arguments)
        _assert_error(result, fragment)

    long_text = "x" * 9000
    bounded = await _run(env, "text_transform", {
        "text": long_text, "operation": "regex_replace",
        "pattern": "x", "replacement": "y",
    })
    _assert_error(bounded, "8192 byte limit")
    unbounded = await _run(env, "text_transform", {"text": long_text, "operation": "upper"})
    assert unbounded.success and unbounded.output["length"] == 9000


# --------------------------------------------------------------------------- base64_codec


async def test_base64_codec_exact_outputs(env):
    result = await _run(env, "base64_codec", {
        "operation": "encode", "text": "hello world",
    })
    assert result.success and result.decision == "allowed"
    assert result.output == {
        "operation": "encode", "text": "aGVsbG8gd29ybGQ=", "length": 16,
    }

    result = await _run(env, "base64_codec", {
        "operation": "decode", "text": "aGVsbG8gd29ybGQ=",
    })
    assert result.output == {
        "operation": "decode", "text": "hello world", "length": 11,
    }

    result = await _run(env, "base64_codec", {
        "operation": "encode", "text": "࠾",
    })
    assert result.output["text"] == "4KC+"

    result = await _run(env, "base64_codec", {
        "operation": "encode", "text": "࠾", "urlsafe": True,
    })
    assert result.output["text"] == "4KC-"

    result = await _run(env, "base64_codec", {
        "operation": "decode", "text": "4KC-", "urlsafe": True,
    })
    assert result.output == {
        "operation": "decode", "text": "࠾", "length": 1,
    }


async def test_base64_codec_malformed_input_errors(env):
    cases = [
        ({"operation": "decode", "text": "not base64!"}, "not valid base64"),
        ({"operation": "decode", "text": "/w=="}, "not valid UTF-8"),
        ({"operation": "decode", "text": "aGVsbG8"}, "not valid base64"),
        ({"operation": "encode", "text": "x", "urlsafe": "yes"}, "invalid arguments"),
        ({"operation": "hash", "text": "x"}, "invalid arguments"),
        ({"operation": "encode"}, "invalid arguments"),
        ({"operation": "decode", "text": ""}, "invalid arguments"),
    ]
    for arguments, fragment in cases:
        result = await _run(env, "base64_codec", arguments)
        _assert_error(result, fragment)


async def test_audit_records_successful_and_failed_data_calls(env):
    ok = await _run(env, "json_query", {"data": "{\"a\": 1}", "path": "a"})
    bad = await _run(env, "csv_parse", {"text": "a,a\n1,2\n"})
    assert ok.success and bad.success is False

    rows = await db.fetchall(
        "SELECT decision, success FROM tool_calls WHERE tool IN "
        "('json_query','csv_parse') ORDER BY created_at, tool"
    )
    assert len(rows) == 2
    assert {row["decision"] for row in rows} == {"allowed", "error"}

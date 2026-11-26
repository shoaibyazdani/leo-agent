"""Delivery token parsing and resolution shared by preview/cards and runtime.

One resolver produces a :class:`DeliveryPlan` for both renderers and the
scheduler. The plan preserves the raw tokens for editing, exposes resolved
:class:`~leo.schedules.domain.DeliveryTarget` objects for execution and gives
honest audience labels (including "no destination") for consent text.

Final-head forms implemented: omitted/empty, ``origin``, ``all``, ``local``,
``bot-chat[:<profile>]``, ``room:<id>[@<profile>]``, a bare configured platform,
``platform:chat[:thread]``, comma strings and lists.

LEO-native extensions (clearly labelled in ``resolved_from``): ``home``,
``local:<id>``, and validated target objects. Unknown tokens and unresolved
machine-local destinations are rejected before persistence; a *known* platform
without a configured home channel resolves to an honest nowhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .domain import DeliveryTarget

# LEO's registered channel adapters plus the three routing pseudo-platforms.
# Kept static so importing the resolver never imports the adapter registry or
# touches the network.
_KNOWN_PLATFORMS: frozenset[str] = frozenset(
    {
        "a2a",
        "bluebubbles",
        "bot-chat",
        "dingtalk",
        "discord",
        "email",
        "feishu",
        "google_chat",
        "irc",
        "line",
        "local",
        "matrix",
        "mattermost",
        "msteams",
        "ntfy",
        "qqbot",
        "relay",
        "room",
        "signal",
        "slack",
        "sms",
        "telegram",
        "wecom",
        "weixin",
        "whatsapp",
    }
)

# On these platforms a colon belongs to the chat/room id itself (Matrix room
# ids look like ``!ops:example.org``); there is no separate thread component to
# split off. Everything else keeps the historical ``chat[:thread]`` grammar.
_COLON_CHAT_PLATFORMS: frozenset[str] = frozenset({"matrix"})

_ROOM_RE = re.compile(
    r"^(?P<room>[A-Za-z0-9][A-Za-z0-9._:-]*)(?:@(?P<profile>[A-Za-z0-9_-]+))?$"
)
_CONTROL_RE = re.compile(
    r"[\x00-\x1f\x7f-\x9f​-‏‪-‮⁠-⁩﻿]"
)
_OBJECT_KEYS = frozenset(
    {
        "token",
        "kind",
        "platform",
        "chat",
        "chat_id",
        "thread",
        "thread_id",
        "profile",
        "room_id",
        "id",
        "conversation_id",
        "label",
    }
)


class DeliveryResolutionError(ValueError):
    """A stable delivery rejection (never a guessed recipient)."""

    def __init__(self, code: str, message: str, *, token: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.token = token


def _fail(code: str, message: str, token: str = "") -> DeliveryResolutionError:
    return DeliveryResolutionError(code, message, token=token)


def _split_target(platform: str, rest: str) -> tuple[str, str | None]:
    """Split ``chat[:thread]`` using per-platform id rules, not every colon."""
    if platform.strip().lower() in _COLON_CHAT_PLATFORMS:
        return rest.strip(), None
    if ":" in rest:
        chat_id, thread_id = rest.split(":", 1)
        thread = thread_id.strip()
        return chat_id.strip(), thread or None
    return rest.strip(), None


def _parse_home_channel(text: str) -> HomeChannel | None:
    """Parse one configured home channel without truncating colon bearing ids."""
    platform, separator, rest = text.partition(":")
    platform = platform.strip().lower()
    if not separator or not platform or not rest.strip():
        return None
    chat_id, thread_id = _split_target(platform, rest)
    if not chat_id:
        return None
    return HomeChannel(platform=platform, chat_id=chat_id, thread_id=thread_id)


@dataclass(frozen=True)
class HomeChannel:
    """A platform's configured home destination, resolved at fire time."""

    platform: str
    chat_id: str
    thread_id: str | None = None


@dataclass(frozen=True)
class Origin:
    """The conversation a create/edit came from."""

    platform: str
    chat_id: str
    thread_id: str | None = None


@dataclass(frozen=True)
class RoomInfo:
    """A machine-local room known to a room-capable adapter."""

    id: str
    profile: str | None = None


@dataclass(frozen=True)
class DeliveryContext:
    """Everything resolution needs; all resolution state is injected.

    ``connected_platforms``/``rooms``/``bot_profiles`` may be ``None`` when the
    caller has no authoritative roster. Named bot profiles and rooms then fail
    closed rather than promising a destination that may not exist.
    """

    origin: Origin | None = None
    home_channels: tuple[HomeChannel, ...] = ()
    connected_platforms: frozenset[str] | None = None
    bot_profiles: frozenset[str] | None = None
    rooms: Mapping[str, RoomInfo] | None = None
    room_capable: bool = False
    default_room_channel: str | None = None
    local_conversations: frozenset[str] | None = None
    known_platforms: frozenset[str] | None = None

    def home_for(self, platform: str) -> HomeChannel | None:
        wanted = platform.strip().lower()
        for home in self.home_channels:
            if home.platform.strip().lower() == wanted and home.chat_id:
                return home
        return None

    def platform_is_connected(self, platform: str) -> bool:
        if self.connected_platforms is None:
            return True
        return platform.strip().lower() in self.connected_platforms


@dataclass(frozen=True)
class DeliveryRoute:
    """One resolved destination plus preview provenance."""

    target: DeliveryTarget
    token: str
    audience: str
    resolved_from: str

    @property
    def identity(self) -> tuple[str, str | None, str | None]:
        return self.target.identity()


@dataclass(frozen=True)
class UnresolvedRoute:
    """A token that names no verifiable destination."""

    token: str
    code: str
    reason: str


@dataclass(frozen=True)
class DeliveryPlan:
    """Preview and runtime view of one delivery lane."""

    raw: str
    lane: str
    routes: tuple[DeliveryRoute, ...] = ()
    unresolved: tuple[UnresolvedRoute, ...] = ()
    local_only: bool = False
    lane_note: str | None = None

    @property
    def targets(self) -> tuple[DeliveryTarget, ...]:
        return tuple(route.target for route in self.routes)

    @property
    def identities(self) -> tuple[tuple[str, str | None, str | None], ...]:
        return tuple(route.identity for route in self.routes)

    @property
    def keyed_targets(self) -> dict[str, DeliveryTarget]:
        """Targets keyed by ``platform:chat:thread`` for runtime receipts."""

        return {self.identity_key(target): target for target in self.targets}

    @staticmethod
    def identity_key(target: DeliveryTarget) -> str:
        """The durable receipt key for one target (never the literal None)."""

        platform, chat, thread = target.identity()
        return ":".join(
            [platform or "nowhere", "" if chat is None else str(chat), "" if thread is None else str(thread)]
        )

    @property
    def nowhere(self) -> bool:
        return not self.routes and not self.local_only

    def audiences(self) -> tuple[str, ...]:
        seen: list[str] = []
        for route in self.routes:
            if route.audience not in seen:
                seen.append(route.audience)
        return tuple(seen)

    def describe(self) -> str:
        if self.local_only:
            extra = [route.audience for route in self.routes if route.audience != "local history only"]
            return "local history only" + (f"; {', '.join(extra)}" if extra else "")
        if self.nowhere:
            return "no destination"
        return ", ".join(self.audiences())

    def is_origin_only(self) -> bool:
        routes = [route for route in self.routes if route.target.kind != "local"]
        return bool(routes) and all(route.resolved_from == "origin" for route in routes)

    @classmethod
    def configured(
        cls,
        targets: Sequence[DeliveryTarget],
        *,
        raw: str = "",
        lane: str = "result",
        lane_note: str | None = None,
    ) -> "DeliveryPlan":
        """Project already-persisted targets without re-resolving tokens."""

        routes: list[DeliveryRoute] = []
        local_only = False
        seen: set[tuple[str, str | None, str | None]] = set()
        for target in targets:
            _reject_placeholder_address(target)
            if target.identity() in seen:
                continue
            seen.add(target.identity())
            local_only = local_only or target.kind == "local"
            audience = target.label or _configured_audience(target)
            routes.append(
                DeliveryRoute(
                    target=target,
                    token=_target_token(target),
                    audience=audience,
                    resolved_from="configured",
                )
            )
        return cls(
            raw=raw,
            lane=lane,
            routes=tuple(routes),
            local_only=local_only,
            lane_note=lane_note,
        )


def _configured_audience(target: DeliveryTarget) -> str:
    if target.kind == "local":
        return "local history only"
    if target.kind == "bot_chat":
        return "local agent chat" if not target.profile else f"local agent chat ({target.profile})"
    if target.kind == "room":
        return f"room {target.room_id or target.chat_id}"
    platform = target.platform
    if target.thread_id:
        return f"{platform} chat {target.chat_id} thread {target.thread_id}"
    return f"{platform} chat {target.chat_id}"


def _target_token(target: DeliveryTarget) -> str:
    if target.kind == "local":
        return "local" if not target.chat_id else f"local:{target.chat_id}"
    if target.kind == "bot_chat":
        return "bot-chat" if not target.profile else f"bot-chat:{target.profile}"
    if target.kind == "room":
        room = target.room_id or target.chat_id or ""
        return f"room:{room}" + (f"@{target.profile}" if target.profile else "")
    if target.chat_id and target.thread_id:
        return f"{target.platform}:{target.chat_id}:{target.thread_id}"
    if target.chat_id:
        return f"{target.platform}:{target.chat_id}"
    return target.platform


def _clean_token(token: Any) -> str:
    if not isinstance(token, str):
        raise _fail("invalid_target", "Delivery targets must be strings or validated objects.")
    return token.strip()


def _reject_controls(token: str) -> None:
    if _CONTROL_RE.search(token):
        raise _fail("invalid_target", "Delivery target contains control characters.")


def _reject_placeholder_address(target: DeliveryTarget) -> None:
    """The string None/null is not an address; never hand it to an adapter."""

    for value in (target.chat_id, target.thread_id):
        if isinstance(value, str) and value.strip().lower() in ("none", "null"):
            raise _fail(
                "invalid_target",
                "Delivery target has a placeholder address (None/null).",
                _target_token(target),
            )


def _normalize_identity(target: DeliveryTarget) -> tuple[str, str | None, str | None]:
    return target.identity()


class DeliveryResolver:
    """The single preview/runtime delivery resolver."""

    def __init__(self, context: DeliveryContext | None = None) -> None:
        self.context = context or DeliveryContext()
        self._known = self.context.known_platforms or _KNOWN_PLATFORMS

    # -- public API -------------------------------------------------------
    def resolve(
        self,
        value: Any,
        *,
        lane: str = "result",
        mode: str = "create",
        strict: bool = True,
    ) -> DeliveryPlan:
        """Resolve a ``deliver``/``failure_deliver`` value into a plan.

        ``mode="create"`` treats an omitted value as origin-with-local-fallback;
        ``mode="runtime"`` treats an omitted/empty value as local.
        """

        tokens, object_tokens = self._normalize_input(value, mode=mode)
        raw = ",".join(
            [token for token, _source in tokens]
            + [token for token, _data in object_tokens]
        )
        routes: list[DeliveryRoute] = []
        unresolved: list[UnresolvedRoute] = []
        local_only = False

        def add_route(route: DeliveryRoute) -> None:
            _reject_placeholder_address(route.target)
            if route.identity in {item.identity for item in routes}:
                return
            routes.append(route)

        def add_unresolved(token: str, code: str, reason: str) -> None:
            # "nowhere" is an honest, allowed state (no captured origin and no
            # configured home). Unknown/unverifiable tokens are hard errors.
            if strict and code != "nowhere":
                raise _fail(code, reason, token)
            unresolved.append(UnresolvedRoute(token=token, code=code, reason=reason))

        for token, source in tokens:
            lowered = token.lower()
            if token == "":
                continue
            # Case-sensitive final-head keywords: "origin"/"local". Everything
            # that is case-insensitive is matched on ``lowered``.
            if token == "origin":
                route = self._resolve_origin(token)
                if route is None:
                    add_unresolved(token, "nowhere", "no captured origin and no configured home channel")
                else:
                    add_route(route)
                continue
            if token == "local":
                local_only = True
                add_route(
                    DeliveryRoute(
                        target=DeliveryTarget(kind="local", platform="local"),
                        token=token,
                        audience="local history only",
                        resolved_from="local",
                    )
                )
                continue
            if lowered == "all":
                homes = self._all_homes()
                if not homes:
                    add_unresolved(token, "nowhere", "no connected platform has a configured home channel")
                for home in homes:
                    add_route(self._route_from_home(home, token, "home"))
                continue
            if lowered == "home":
                home = self._first_home()
                if home is None:
                    add_unresolved(token, "nowhere", "no configured home channel")
                else:
                    add_route(self._route_from_home(home, token, "home"))
                continue
            if lowered == "bot-chat" or lowered.startswith("bot-chat:"):
                add_route(self._resolve_bot_chat(token))
                continue
            if lowered.startswith("room:"):
                route = self._resolve_room(token)
                if route is None:
                    add_unresolved(token, "unresolved_room", "room does not resolve on this machine")
                else:
                    add_route(route)
                continue
            if lowered.startswith("local:"):
                route = self._resolve_local_conversation(token)
                local_only = True
                add_route(route)
                continue
            if ":" in token:
                route = self._resolve_explicit(token)
                if route is None:
                    add_unresolved(token, "nowhere", "no destination for this chat")
                else:
                    add_route(route)
                continue
            # A bare platform name means that platform's home channel, with a
            # same-platform origin fallback when no home is configured.
            route = self._resolve_bare_platform(token)
            if route is None:
                add_unresolved(token, "nowhere", f"{token!r} has no configured home channel")
            else:
                add_route(route)

        for token, data in object_tokens:
            for route in self._resolve_object(token, data):
                if route.target.kind == "local":
                    local_only = True
                add_route(route)

        if not routes and not unresolved:
            # No usable destination at all: fail closed on create, but a stored
            # legacy row may still be honestly nowhere at runtime.
            if mode == "create":
                add_unresolved(raw or "origin", "nowhere", "delivery resolved to no destination")

        return DeliveryPlan(
            raw=raw,
            lane=lane,
            routes=tuple(routes),
            unresolved=tuple(unresolved),
            local_only=local_only,
        )

    def resolve_result(self, value: Any, *, mode: str = "create", strict: bool = True) -> DeliveryPlan:
        return self.resolve(value, lane="result", mode=mode, strict=strict)

    def resolve_failure(self, value: Any, *, mode: str = "create", strict: bool = True) -> DeliveryPlan:
        return self.resolve(value, lane="failure", mode=mode, strict=strict)

    def resolve_stored(
        self, value: Any, *, lane: str = "result", strict: bool = False
    ) -> DeliveryPlan:
        """Resolve an already-persisted target list through the same resolver.

        Runtime never reinterprets card tokens: it reads the validated targets
        that were committed with the schedule and projects them into the same
        :class:`DeliveryPlan` shape the preview used. Legacy payloads that still
        store raw tokens go through the token resolver instead, honestly and
        without inventing a destination.
        """
        if isinstance(value, (list, tuple)) and value and all(
            isinstance(item, Mapping)
            and set(item).issubset(DeliveryTarget.model_fields)
            and item.get("platform")
            for item in value
        ):
            try:
                targets = [DeliveryTarget.model_validate(dict(item)) for item in value]
            except Exception as exc:  # noqa: BLE001 - a corrupt row must fail closed
                raise _fail("invalid_target", f"stored delivery target is invalid: {exc}") from None
            return DeliveryPlan.configured(targets, lane=lane)
        if isinstance(value, (list, tuple)) and not value:
            return DeliveryPlan(raw="", lane=lane)
        return self.resolve(value, lane=lane, mode="runtime", strict=strict)

    # -- input normalization ---------------------------------------------
    def _normalize_input(self, value: Any, *, mode: str) -> tuple[list[tuple[str, str]], list[tuple[str, Any]]]:
        tokens: list[tuple[str, str]] = []
        objects: list[tuple[str, Any]] = []
        if value is None or (isinstance(value, str) and not value.strip()) or value == [] or value == ():
            if mode == "runtime":
                return [("local", "plain")], objects
            # Omitted create delivery: origin when capturable, else local. A
            # token explicitly written as "origin" still takes the home
            # fallback; only an *omitted* value collapses to local.
            origin = self.context.origin
            if origin is not None and origin.platform and origin.chat_id:
                return [("origin", "plain")], objects
            return [("local", "plain")], objects
        if isinstance(value, str):
            for part in value.split(","):
                token = part.strip()
                tokens.append((token, "comma"))
            return tokens, objects
        if isinstance(value, Mapping):
            token = self._object_token(value)
            objects.append((token, dict(value)))
            return tokens, objects
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    token = self._object_token(item)
                    objects.append((token, dict(item)))
                else:
                    text = _clean_token(item)
                    for part in text.split(","):
                        tokens.append((part.strip(), "list"))
            return tokens, objects
        raise _fail("invalid_target", "delivery must be a string, list, or validated object.")

    def _object_token(self, value: Mapping[str, Any]) -> str:
        unknown = set(value) - _OBJECT_KEYS
        if unknown:
            raise _fail("invalid_target", f"Unknown delivery target key(s): {', '.join(sorted(unknown))}.")
        if "token" in value:
            token = _clean_token(value["token"])
            if not token:
                raise _fail("invalid_target", "Delivery object token must not be empty.")
            _reject_controls(token)
            return token
        kind = str(value.get("kind") or "").strip().lower()
        if kind == "local":
            identifier = value.get("conversation_id") or value.get("chat_id") or value.get("id")
            return "local" if not identifier else f"local:{identifier}"
        if kind == "bot_chat" or kind == "bot-chat":
            profile = value.get("profile")
            if profile in (None, ""):
                return "bot-chat"
            return f"bot-chat:{profile}"
        if kind == "room":
            room = value.get("room_id") or value.get("chat_id") or value.get("id")
            if not room:
                raise _fail("invalid_target", "Room delivery object requires room_id.")
            profile = value.get("profile")
            return f"room:{room}" + (f"@{profile}" if profile else "")
        if kind and kind not in ("external", "chat"):
            raise _fail("invalid_target", f"Unknown delivery target kind {kind!r}.")
        platform = value.get("platform")
        chat = value.get("chat") or value.get("chat_id")
        if not platform or chat in (None, ""):
            raise _fail("invalid_target", "Delivery target requires platform and chat.")
        thread = value.get("thread") or value.get("thread_id")
        token = f"{platform}:{chat}" + (f":{thread}" if thread not in (None, "") else "")
        _reject_controls(str(token))
        return str(token)

    def _token_label(self, token: str, data: Mapping[str, Any]) -> str:
        label = data.get("label")
        if not label:
            return token
        text = str(label).strip()
        if not text:
            return token
        _reject_controls(text)
        if len(text) > 200:
            raise _fail("invalid_target", "Delivery target label is too long.", token)
        return text

    # -- individual token resolution -------------------------------------
    def _resolve_origin(self, token: str) -> DeliveryRoute | None:
        origin = self.context.origin
        if origin is not None and origin.platform and origin.chat_id:
            platform = origin.platform.strip().lower()
            if not self.context.platform_is_connected(platform):
                return None
            target = DeliveryTarget(
                kind="external",
                platform=platform,
                chat_id=str(origin.chat_id),
                thread_id=str(origin.thread_id) if origin.thread_id is not None else None,
                label="this chat",
            )
            return DeliveryRoute(target=target, token=token, audience="this chat", resolved_from="origin")
        home = self._first_home()
        if home is None:
            return None
        return self._route_from_home(home, token, "origin_fallback")

    def _first_home(self) -> HomeChannel | None:
        for home in self.context.home_channels:
            if not home.chat_id:
                continue
            if self.context.platform_is_connected(home.platform):
                return home
        return None

    def _all_homes(self) -> list[HomeChannel]:
        homes: list[HomeChannel] = []
        seen: set[tuple[str, str, str | None]] = set()
        for home in self.context.home_channels:
            if not home.chat_id:
                continue
            platform = home.platform.strip().lower()
            if platform in ("bot-chat", "room", "local"):
                continue
            if not self.context.platform_is_connected(platform):
                continue
            identity = (platform, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
            if identity in seen:
                continue
            seen.add(identity)
            homes.append(home)
        return homes

    def _route_from_home(self, home: HomeChannel, token: str, resolved_from: str) -> DeliveryRoute:
        platform = home.platform.strip().lower()
        target = DeliveryTarget(
            kind="external",
            platform=platform,
            chat_id=str(home.chat_id),
            thread_id=str(home.thread_id) if home.thread_id is not None else None,
        )
        audience = "this chat" if resolved_from == "origin" else f"home channel for {platform}"
        return DeliveryRoute(target=target, token=token, audience=audience, resolved_from=resolved_from)

    def _resolve_bot_chat(self, token: str) -> DeliveryRoute:
        lowered = token.lower()
        if lowered == "bot-chat":
            target = DeliveryTarget(kind="bot_chat", platform="bot-chat", chat_id="", profile=None)
            return DeliveryRoute(
                target=target, token=token, audience="local agent chat", resolved_from="bot_chat"
            )
        profile = token.split(":", 1)[1].strip()
        if not profile:
            target = DeliveryTarget(kind="bot_chat", platform="bot-chat", chat_id="", profile=None)
            return DeliveryRoute(
                target=target, token=token, audience="local agent chat", resolved_from="bot_chat"
            )
        _reject_controls(profile)
        if self.context.bot_profiles is None or profile not in self.context.bot_profiles:
            raise _fail(
                "unknown_bot_profile",
                f"bot-chat profile {profile!r} is not available on this machine.",
                token,
            )
        target = DeliveryTarget(kind="bot_chat", platform="bot-chat", chat_id=profile, profile=profile)
        return DeliveryRoute(
            target=target,
            token=token,
            audience=f"local agent chat ({profile})",
            resolved_from="bot_chat",
        )

    def _resolve_room(self, token: str) -> DeliveryRoute | None:
        if not self.context.room_capable:
            raise _fail("room_unavailable", "This machine has no room-capable adapter.", token)
        address = token.split(":", 1)[1].strip()
        match = _ROOM_RE.fullmatch(address)
        if match is None:
            raise _fail(
                "unresolved_room",
                "Malformed room target; expected room:<id>[@<profile>].",
                token,
            )
        room_id = match.group("room")
        profile = match.group("profile")
        if profile and (self.context.bot_profiles is None or profile not in self.context.bot_profiles):
            raise _fail(
                "unknown_bot_profile",
                f"room profile {profile!r} is not available on this machine.",
                token,
            )
        known_room = False
        if self.context.rooms is not None and room_id in self.context.rooms:
            info = self.context.rooms[room_id]
            if profile and info.profile and info.profile != profile:
                raise _fail("unresolved_room", "Room profile does not match its hosting profile.", token)
            known_room = True
        if self.context.default_room_channel and room_id == self.context.default_room_channel:
            known_room = True
        if not known_room:
            raise _fail("unresolved_room", "No live room with that id on this machine.", token)
        target = DeliveryTarget(
            kind="room",
            platform="room",
            chat_id=room_id,
            thread_id=None,
            room_id=room_id,
            profile=profile,
        )
        audience = f"room {room_id}" + (f" on profile {profile}" if profile else "")
        return DeliveryRoute(target=target, token=token, audience=audience, resolved_from="room")

    def _resolve_local_conversation(self, token: str) -> DeliveryRoute:
        identifier = token.split(":", 1)[1].strip()
        if not identifier:
            raise _fail("invalid_target", "local:<id> requires a conversation id.", token)
        _reject_controls(identifier)
        if (
            self.context.local_conversations is not None
            and identifier not in self.context.local_conversations
        ):
            raise _fail(
                "unknown_local_conversation",
                f"local conversation {identifier!r} does not exist.",
                token,
            )
        target = DeliveryTarget(kind="local", platform="local", chat_id=identifier)
        return DeliveryRoute(
            target=target,
            token=token,
            audience=f"local conversation {identifier}",
            resolved_from="local_conversation",
        )

    def _validated_platform(self, platform_text: Any, token: str) -> str:
        platform = str(platform_text).strip().lower()
        if not platform or subplatform_invalid(platform):
            raise _fail("invalid_target", "Delivery target has an empty or invalid platform.", token)
        if platform not in self._known:
            raise _fail("unknown_token", f"Unknown delivery platform {platform_text!r}.", token)
        if not self.context.platform_is_connected(platform):
            raise _fail(
                "adapter_unavailable",
                f"The {platform!r} adapter is not connected; delivery cannot be verified.",
                token,
            )
        return platform

    def _build_external_route(
        self, platform: str, chat_id: Any, thread_id: Any, token: str
    ) -> DeliveryRoute:
        address = str(chat_id).strip()
        thread = str(thread_id).strip() if thread_id not in (None, "") else None
        if not address:
            raise _fail("invalid_target", "Delivery target requires a chat id.", token)
        _reject_controls(address)
        if thread:
            _reject_controls(thread)
        origin = self.context.origin
        # Chat-only explicit target on the origin chat adopts the origin thread
        # only where a thread is a durable place (Telegram/Slack), as final head
        # does.
        if (
            thread is None
            and platform in ("telegram", "slack")
            and origin is not None
            and origin.platform.strip().lower() == platform
            and str(origin.chat_id) == address
            and origin.thread_id
        ):
            thread = str(origin.thread_id)
        target = DeliveryTarget(
            kind="external",
            platform=platform,
            chat_id=address,
            thread_id=thread,
        )
        audience = f"{platform} chat {address}" + (f" thread {thread}" if thread else "")
        return DeliveryRoute(target=target, token=token, audience=audience, resolved_from="explicit")

    def _resolve_explicit(self, token: str) -> DeliveryRoute | None:
        platform_text, rest = token.split(":", 1)
        platform = self._validated_platform(platform_text, token)
        rest = rest.strip()
        if not rest:
            raise _fail("invalid_target", "Delivery target requires a chat id.", token)
        chat_id, thread_id = _split_target(platform, rest)
        return self._build_external_route(platform, chat_id, thread_id, token)

    def _resolve_external_object(self, data: Mapping[str, Any], token: str) -> DeliveryRoute:
        """Resolve a structured target directly: no lossy token re-parse."""
        platform = self._validated_platform(data.get("platform"), token)
        chat_id = data.get("chat") or data.get("chat_id")
        if chat_id in (None, ""):
            raise _fail("invalid_target", "Delivery target requires a chat id.", token)
        thread_id = data.get("thread") or data.get("thread_id")
        return self._build_external_route(
            platform, str(chat_id), thread_id, token
        )

    def _resolve_bare_platform(self, token: str) -> DeliveryRoute | None:
        platform = token.strip().lower()
        _reject_controls(platform)
        if subplatform_invalid(platform):
            raise _fail("invalid_target", "Delivery target has an invalid platform.", token)
        if platform not in self._known:
            raise _fail("unknown_token", f"Unknown delivery target {token!r}.", token)
        if not self.context.platform_is_connected(platform):
            raise _fail(
                "adapter_unavailable",
                f"The {platform!r} adapter is not connected; delivery cannot be verified.",
                token,
            )
        home = self.context.home_for(platform)
        if home is not None:
            return self._route_from_home(home, token, "home")
        origin = self.context.origin
        if origin is not None and origin.platform.strip().lower() == platform and origin.chat_id:
            target = DeliveryTarget(
                kind="external",
                platform=platform,
                chat_id=str(origin.chat_id),
                thread_id=str(origin.thread_id) if origin.thread_id is not None else None,
            )
            return DeliveryRoute(
                target=target, token=token, audience="this chat", resolved_from="origin"
            )
        return None

    def _resolve_object(self, token: str, data: Mapping[str, Any]) -> list[DeliveryRoute]:
        lowered = token.lower()
        routes: list[DeliveryRoute] = []
        if (
            "token" not in data
            and data.get("platform")
            and (data.get("chat") or data.get("chat_id")) not in (None, "")
            and str(data.get("kind") or "external").strip().lower()
            in ("", "external", "chat")
        ):
            # Structured targets never go through colon-delimited tokens: a
            # Matrix room id such as !ops:example.org stays one chat id.
            routes.append(self._resolve_external_object(data, token))
        elif token == "origin":
            route = self._resolve_origin(token)
            if route is None:
                raise _fail("nowhere", "No captured origin and no configured home channel.", token)
            routes.append(route)
        elif token == "local" or lowered.startswith("local:"):
            routes.append(
                self._resolve_local_conversation(token)
                if lowered.startswith("local:")
                else DeliveryRoute(
                    target=DeliveryTarget(kind="local", platform="local"),
                    token=token,
                    audience="local history only",
                    resolved_from="local",
                )
            )
        elif lowered == "all":
            homes = self._all_homes()
            if not homes:
                raise _fail("nowhere", "No connected platform has a configured home channel.", token)
            routes.extend(self._route_from_home(home, token, "home") for home in homes)
        elif lowered == "home":
            home = self._first_home()
            if home is None:
                raise _fail("nowhere", "No configured home channel.", token)
            routes.append(self._route_from_home(home, token, "home"))
        elif lowered == "bot-chat" or lowered.startswith("bot-chat:"):
            routes.append(self._resolve_bot_chat(token))
        elif lowered.startswith("room:"):
            route = self._resolve_room(token)
            if route is None:
                raise _fail("unresolved_room", "Room does not resolve on this machine.", token)
            routes.append(route)
        elif ":" in token:
            route = self._resolve_explicit(token)
            if route is None:
                raise _fail("nowhere", f"{token!r} has no destination.", token)
            routes.append(route)
        else:
            route = self._resolve_bare_platform(token)
            if route is None:
                raise _fail("nowhere", f"{token!r} has no configured home channel.", token)
            routes.append(route)
        label = self._token_label(token, data)
        return [
            DeliveryRoute(
                target=item.target,
                token=item.token,
                audience=label,
                resolved_from=item.resolved_from,
            )
            for item in routes
        ]


def subplatform_invalid(platform: str) -> bool:
    return bool(re.search(r"[\s\x00-\x1f\x7f]", platform))


def delivery_context_from_settings(
    settings: Any,
    *,
    origin: Origin | None = None,
    adapters: Mapping[str, Any] | None = None,
) -> DeliveryContext:
    """Build the LEO-native delivery context from operator-owned settings.

    Nothing here is inferred from a request body: home channels, named local
    conversations and the default room come from configuration, and room
    capability comes from the registered adapter objects/classes.
    """
    automation = getattr(settings, "automation", None)
    homes: list[HomeChannel] = []
    for raw in list(getattr(automation, "home_channels", None) or []):
        home = _parse_home_channel(str(raw).strip())
        if home is not None:
            homes.append(home)
    room_capable = False
    for adapter in (adapters or {}).values():
        if bool(getattr(adapter, "supports_rooms", False)):
            room_capable = True
            break
    return DeliveryContext(
        origin=origin,
        home_channels=tuple(homes),
        connected_platforms=(
            frozenset(str(key).strip().lower() for key in adapters) if adapters is not None else None
        ),
        bot_profiles=frozenset(str(item) for item in (getattr(automation, "bot_profiles", None) or [])),
        local_conversations=frozenset(
            str(item) for item in (getattr(automation, "local_conversations", None) or [])
        ),
        room_capable=room_capable,
        default_room_channel=getattr(automation, "default_room_channel", None) or None,
    )


def task_delivery_plan(
    payload: Mapping[str, Any] | None,
    *,
    lane: str,
    resolver: DeliveryResolver | None = None,
) -> tuple[DeliveryPlan, bool]:
    """The runtime view of one schedule payload's delivery lane.

    The second value says whether that lane was explicitly configured. An
    unconfigured failure lane follows the result targets, matching the compiler's
    preview; nowhere is returned honestly rather than replaced by a guess.
    """
    data = payload if isinstance(payload, Mapping) else {}
    result = data.get("delivery")
    failure = data.get("failure_delivery")
    explicit = bool(failure) if lane == "failure" else True
    value = failure if lane == "failure" and failure else result
    plan = (resolver or DeliveryResolver()).resolve_stored(value, lane=lane)
    return plan, explicit


def resolve_delivery(
    value: Any,
    *,
    context: DeliveryContext | None = None,
    lane: str = "result",
    mode: str = "create",
    strict: bool = True,
) -> DeliveryPlan:
    """Convenience wrapper around :class:`DeliveryResolver`."""

    return DeliveryResolver(context).resolve(value, lane=lane, mode=mode, strict=strict)


__all__ = [
    "DeliveryContext",
    "DeliveryPlan",
    "DeliveryResolutionError",
    "DeliveryResolver",
    "DeliveryRoute",
    "HomeChannel",
    "Origin",
    "RoomInfo",
    "UnresolvedRoute",
    "delivery_context_from_settings",
    "resolve_delivery",
    "task_delivery_plan",
]

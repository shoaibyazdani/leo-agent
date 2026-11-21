"""Static contracts for the shared LEO UI motion layer.

The dashboard and Agent Activity are plain browser CSS/JS. These tests read the
shipped stylesheet and assert the HIG motion contract: token-only 200-300 ms
durations, spring curves, one-shot keyframes for every state change,
interruptible drawer/disclosure transitions, no decorative repetitions, and
accessible Reduce Motion / Increase Contrast paths.

This file checks motion.css plus the static integration contracts: both pages
must load it after their own stylesheets, and chat.css's mobile conversation
drawer must keep its closed-state transform under reduced motion.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "leo" / "interfaces" / "static"
MOTION_CSS_PATH = STATIC / "ui" / "motion.css"
CSS = MOTION_CSS_PATH.read_text(encoding="utf-8")
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")
AGENTS_LIVE_HTML = (STATIC / "agents-live.html").read_text(encoding="utf-8")
CHAT_CSS = (STATIC / "ui" / "conversation" / "chat.css").read_text(encoding="utf-8")


def rule(source: str, selector: str) -> str:
    """Return the full CSS rule (including nested blocks) for an exact selector."""
    match = re.search(
        r"(?m)^" + re.escape(selector) + r"\s*\{",
        source,
    )
    assert match, f"missing rule for {selector!r}"
    start = match.start()
    brace = source.index("{", match.end() - 1)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unclosed rule for {selector!r}")


def at_rule(source: str, header: str) -> str:
    """Return a complete at-rule block, e.g. '@media (prefers-reduced-motion: reduce)'."""
    match = re.search(r"(?m)^" + re.escape(header) + r"\s*\{", source)
    assert match, f"missing at-rule {header!r}"
    brace = source.index("{", match.end() - 1)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[match.start():index + 1]
    raise AssertionError(f"unclosed at-rule {header!r}")


def keyframe_bodies(source: str) -> dict[str, str]:
    return {
        name: body
        for name, body in re.findall(
            r"(?ms)^@keyframes\s+([\w-]+)\s*\{(.*?)^\}", source
        )
    }


def declarations(source: str, properties: tuple[str, ...]) -> list[tuple[str, str]]:
    return [
        (name, value.strip())
        for name, value in re.findall(r"(?m)^\s*([\w-]+)\s*:\s*([^;]+);", source)
        if name in properties
    ]


def matching_brace(source: str, opening: int) -> int:
    """Return the index of the brace closing the block that opens at `opening`."""
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("unclosed CSS block")


def style_rules(source: str) -> list[tuple[str, str]]:
    """Return (selector, body) for every style rule in a CSS fragment.

    Comments and at-rule wrappers are ignored; keyframe percentage rules are
    not style rules. Pass the *body* of an at-rule (not its header) to inspect
    rules inside a media query.
    """
    rules: list[tuple[str, str]] = []
    # Braces inside comments (e.g. `.chat-sessions { transform: none }`) are
    # not rule bodies; strip comments before scanning for braces.
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    position = 0
    while True:
        opening = source.find("{", position)
        if opening == -1:
            break
        header = re.sub(r"/\*.*?\*/", "", source[position:opening], flags=re.S).strip()
        closing = matching_brace(source, opening)
        body = source[opening + 1:closing]
        if header and not header.startswith("@") and not re.fullmatch(r"\d+(?:\.\d+)?%", header):
            rules.append((header, body))
        position = closing + 1
    return rules


def media_body(header: str) -> str:
    """Return the declarations/rules inside an at-rule, without its header."""
    return at_rule(CSS, header).split("\n", 1)[1]


# ---- token contract ------------------------------------------------------

REQUIRED_TOKENS = (
    "--motion-accent",
    "--motion-accent-soft",
    "--dur-micro",
    "--dur-base",
    "--dur-panel",
    "--dur-reduced",
    "--ease-spring",
    "--ease-spring-out",
    "--ease-crossfade",
    "--dur-fast",
    "--dur-slow",
    "--ease-standard",
    "--ease-enter",
    "--ease-exit",
    "--motion-shift-xs",
    "--motion-shift-sm",
    "--motion-shift-md",
    "--motion-shift-panel",
    "--motion-scale-pop",
    "--motion-progress",
    "--motion-stagger",
    "--motion-index",
    "--motion-dot-size",
    "--motion-no-delay",
    "--motion-one-iteration",
    "--motion-focus-width",
    "--motion-focus-offset",
    "--motion-hc-border-width",
)


@pytest.mark.parametrize("token", REQUIRED_TOKENS)
def test_required_motion_token_is_defined(token: str) -> None:
    assert re.search(r"(?m)^\s*" + re.escape(token) + r"\s*:", CSS), token


@pytest.mark.parametrize("token", ("--dur-micro", "--dur-base", "--dur-panel"))
def test_default_durations_are_200_to_300_ms(token: str) -> None:
    root = rule(CSS, ":root")
    match = re.search(re.escape(token) + r"\s*:\s*(\d+)ms", root)
    assert match, f"{token} must be an explicit millisecond duration"
    assert 200 <= int(match.group(1)) <= 300


def test_reduced_motion_crossfade_token_is_short() -> None:
    root = rule(CSS, ":root")
    match = re.search(r"--dur-reduced\s*:\s*(\d+)ms", root)
    assert match
    assert 0 < int(match.group(1)) <= 200


def test_back_compatible_aliases_point_at_the_new_tokens() -> None:
    root = rule(CSS, ":root")
    for alias, target in (
        ("--dur-fast", "--dur-micro"),
        ("--dur-slow", "--dur-panel"),
        ("--ease-standard", "--ease-spring-out"),
        ("--ease-enter", "--ease-spring"),
        ("--ease-exit", "--ease-spring-out"),
    ):
        assert re.search(
            re.escape(alias) + r"\s*:\s*var\(" + re.escape(target) + r"\)", root
        ), alias


def test_one_accent_drives_motion_and_has_a_fallback() -> None:
    root = rule(CSS, ":root")
    assert re.search(r"--motion-accent\s*:\s*var\(--accent,", root)
    assert re.search(r"--motion-accent-soft\s*:\s*var\(--accent-soft,", root)


# ---- curves, keyframes, repetition --------------------------------------

def test_all_easing_tokens_are_spring_style_cubic_beziers() -> None:
    root = rule(CSS, ":root")
    for name in ("--ease-spring", "--ease-spring-out", "--ease-crossfade", "--ease-press"):
        match = re.search(re.escape(name) + r"\s*:\s*([^;]+);", root)
        assert match, name
        assert re.fullmatch(r"cubic-bezier\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\)", match.group(1)), (name, match.group(1))


def test_no_linear_easing_anywhere() -> None:
    # Standalone 'linear' only: never as a transition/animation timing function.
    assert not re.search(r"(?<![\w-])linear(?![\w-])", CSS, re.IGNORECASE)


def test_no_decorative_repetition() -> None:
    assert not re.search(r"(?<![\w-])infinite(?![\w-])", CSS, re.IGNORECASE)
    assert "iteration-count: var(--motion-one-iteration)" in CSS


REQUIRED_KEYFRAMES = (
    "motion-card-in", "motion-card-out",
    "motion-drawer-in", "motion-drawer-out",
    "motion-toast-in", "motion-toast-out",
    "motion-banner-in", "motion-banner-out",
    "motion-view-in", "motion-view-out",
    "motion-disclosure-in", "motion-disclosure-out",
    "motion-dot-once",
)


@pytest.mark.parametrize("name", REQUIRED_KEYFRAMES)
def test_required_enter_or_exit_keyframe_exists(name: str) -> None:
    assert name in keyframe_bodies(CSS)


def test_views_and_modals_fade_without_travel() -> None:
    bodies = keyframe_bodies(CSS)
    for name in ("motion-view-in", "motion-view-out"):
        assert "opacity" in bodies[name]
        assert "transform" not in bodies[name]


def test_status_dot_has_one_finite_flash() -> None:
    body = keyframe_bodies(CSS)["motion-dot-once"]
    assert "opacity" in body and "transform" in body
    assert "0%" in body and "60%" in body and "100%" in body


def test_every_animation_is_one_shot_with_a_fill_mode() -> None:
    for prop, value in declarations(CSS, ("animation", "animation-duration", "animation-delay", "animation-iteration-count", "transition", "transition-delay")):
        if value == "none" or value.startswith("none !important"):
            continue
        assert "var(--" in value, (prop, value)
        if prop == "animation":
            assert "both" in value.split(), (prop, value)


# ---- no font-metric or geometry animation --------------------------------

FORBIDDEN_ANIMATED_PROPERTIES = re.compile(
    r"\b(?:font-size|font|line-height|letter-spacing|text-size|width|height|"
    r"top|right|bottom|left|margin|padding|grid-template-columns|"
    r"background-position|box-shadow|filter)\b"
)


def test_no_keyframe_animates_text_size_or_geometry() -> None:
    for name, body in keyframe_bodies(CSS).items():
        properties = re.findall(r"(?m)^\s*([\w-]+)\s*:", body)
        offenders = [p for p in properties if FORBIDDEN_ANIMATED_PROPERTIES.search(p)]
        assert not offenders, (name, offenders)


def test_no_transition_animates_text_size_or_geometry() -> None:
    for prop, value in declarations(CSS, ("animation", "transition")):
        if value == "none":
            continue
        assert not FORBIDDEN_ANIMATED_PROPERTIES.search(value), (prop, value)


# ---- generic adoption selectors -----------------------------------------

REQUIRED_SELECTORS = (
    '[data-motion="enter"]',
    '[data-motion="leave"]',
    ".view,",
    ".drawer,",
    ".toast {",
    ".chat-banner,",
    ".chat-tool details,",
    '[data-motion="disclosure"]',
    ".status__dot,",
    ".motion-disclosure {",
    ":focus-visible",
    "font-variant-numeric: tabular-nums",
)


@pytest.mark.parametrize("needle", REQUIRED_SELECTORS)
def test_generic_adoption_hook_exists(needle: str) -> None:
    assert needle in CSS


def test_drawer_is_interruptible_with_transitions_not_keyframes() -> None:
    drawer = rule(CSS, '.drawer, [data-motion="drawer"]')
    assert "transition:" in drawer
    assert "transform var(--dur-panel) var(--ease-spring)" in drawer
    assert "visibility 0s var(--ease-crossfade) var(--dur-panel)" in drawer
    assert "animation:" not in drawer
    opened = rule(CSS, '.drawer:not([hidden]),\n[data-motion="drawer"]:not([hidden]),\n.drawer[data-motion="open"],\n[data-motion="drawer"][data-motion="open"]')
    assert "transform: none" in opened
    assert "transition-delay: var(--motion-no-delay)" in opened


def test_disclosure_supports_details_and_grid_wrapper() -> None:
    assert "grid-template-rows: 1fr" in rule(CSS, ".motion-disclosure")
    assert 'grid-template-rows: 0fr' in rule(CSS, '.motion-disclosure[data-expanded="false"]')
    assert "details[open]" in CSS
    assert "summary::after" in CSS


# ---- accessibility overrides --------------------------------------------

def test_reduce_motion_policy_replaces_travel_with_crossfades() -> None:
    block = at_rule(CSS, "@media (prefers-reduced-motion: reduce)")
    assert "--dur-base: var(--dur-reduced)" in block
    assert "--dur-panel: var(--dur-reduced)" in block
    assert "--ease-spring: var(--ease-crossfade)" in block
    assert "--motion-shift-panel: 0px" in block
    assert "animation-iteration-count: var(--motion-one-iteration) !important" in block
    assert "transition-property: opacity, color, background-color, border-color, outline-color !important" in block
    assert "transform: none !important" in block
    assert "transition: opacity var(--dur-reduced) var(--ease-crossfade) !important" in block
    assert "animation: none !important" in block
    assert ".chat-sessions" in block
    assert ".drawer" in block


def test_increase_contrast_policy_uses_solid_borders_and_no_pulses() -> None:
    block = at_rule(CSS, "@media (prefers-contrast: more)")
    assert "--motion-focus-width: 3px" in block
    assert "--motion-hc-border-width: 2px" in block
    assert "border-color: currentColor" in block
    assert "backdrop-filter: none" in block
    assert "opacity: 1 !important" in block
    assert "animation: none !important" in block


def test_forced_colors_policy_keeps_visible_borders() -> None:
    block = at_rule(CSS, "@media (forced-colors: active)")
    assert "solid CanvasText" in block
    assert "outline-color: Highlight" in block
    assert "animation: none !important" in block


def test_focus_ring_is_an_outline_never_a_glow() -> None:
    focus_rules = [
        body for selector, body in style_rules(CSS)
        if selector.endswith("[tabindex]:focus-visible")
        and "outline: var(--motion-focus-width) solid var(--motion-accent)" in body
    ]
    assert focus_rules
    block = focus_rules[0]
    assert "outline-offset: var(--motion-focus-offset)" in block
    assert "box-shadow: none" in block
    assert "transition: none" in block
    assert "animation: none" in block


# ---- shipped integration contracts (the eight latent accessibility fixes) --

LINK = r'<link\s+rel="stylesheet"\s+href="([^"]+)"\s*/>'


def test_motion_stylesheet_is_linked_after_existing_stylesheets() -> None:
    """Both pages load the motion layer once, after their own stylesheets."""
    for name, source, prereqs in (
        (
            "index.html", INDEX_HTML,
            ("/assets/ui/theme.css", "/assets/ui/components.css",
             "/assets/ui/conversation/chat.css"),
        ),
        ("agents-live.html", AGENTS_LIVE_HTML, ("/assets/ui/agents-live.css",)),
    ):
        # Asset cache-busting queries are not part of the stylesheet identity.
        links = [href.split("?", 1)[0] for href in re.findall(LINK, source)]
        assert links.count("/assets/ui/motion.css") == 1, name
        motion_index = links.index("/assets/ui/motion.css")
        for prereq in prereqs:
            assert prereq in links, (name, prereq)
            assert links.index(prereq) < motion_index, (name, prereq)


def test_reduced_motion_does_not_force_closed_conversation_drawer_open() -> None:
    """chat.css hides the mobile sessions panel with a transform when closed."""
    reduced = media_body("@media (prefers-reduced-motion: reduce)")
    sessions_rules = [(s, d) for s, d in style_rules(reduced) if ".chat-sessions" in s]
    assert sessions_rules, "the open conversation drawer still needs an override"
    for selector, body in sessions_rules:
        assert ".chat-shell.chat-history-open .chat-sessions" in selector, selector
        assert "transform: none !important" in body
    assert not re.search(r"(?m)^\s*\.chat-sessions\s*[,{]", reduced), (
        "a blanket .chat-sessions override permanently exposes the closed drawer"
    )
    # chat.css is loaded before motion.css; its off-screen closed state survives.
    assert "transform: translateX(calc(-100vw" in CHAT_CSS


def test_forced_colors_keeps_a_real_focus_indicator() -> None:
    forced = media_body("@media (forced-colors: active)")
    focus = [(s, d) for s, d in style_rules(forced) if "focus-visible" in s]
    assert len(focus) == 1
    selector, body = focus[0]
    assert ":where(" not in selector, ":where() cannot outrank theme.css outline:none"
    assert re.search(r"outline:\s*var\(--motion-focus-width\) solid Highlight !important", body)
    assert "outline-offset: var(--motion-focus-offset) !important" in body
    assert "outline-color: Highlight !important" in body
    assert "box-shadow: none !important" in body
    # The base focus rule must also outrank theme.css's `:focus-visible` (0,1,0).
    base = [
        s for s, d in style_rules(CSS)
        if "focus-visible" in s
        and "outline: var(--motion-focus-width) solid var(--motion-accent)" in d
    ]
    assert base and all(":where(" not in s for s in base)


def test_progress_fill_uses_the_shipped_value_token() -> None:
    progress = rule(CSS, ".progress__fill")
    assert "width: var(--value, var(--motion-progress))" in progress
    assert "scaleX" not in progress
    assert "transition: width" not in progress
    assert "motion-progress-in" not in CSS
    assert "--motion-progress: 0%" in rule(CSS, ":root")


def test_leave_hooks_still_fire_animationend_under_reduced_motion() -> None:
    reduced = media_body("@media (prefers-reduced-motion: reduce)")
    leave_rules = [(s, d) for s, d in style_rules(reduced) if '[data-motion="leave"]' in s]
    assert leave_rules
    for _, body in leave_rules:
        assert "animation-duration: var(--dur-reduced) !important" in body
        assert "animation-iteration-count: var(--motion-one-iteration) !important" in body
    for header in (
        "@media (prefers-reduced-motion: reduce)",
        "@media (prefers-contrast: more)",
        "@media (forced-colors: active)",
    ):
        for selector, body in style_rules(media_body(header)):
            if re.search(r"(?m)^\s*animation:\s*none", body):
                assert '[data-motion="leave"]' not in selector, selector
                assert ".toast" not in selector, selector
    # CSS cannot fire an event for a node that is hidden or removed; callers
    # must keep a timeout fallback alongside animationend.
    assert "setTimeout fallback" in CSS


def test_collapsed_disclosure_content_is_out_of_the_tab_order() -> None:
    collapsed = [
        (s, d) for s, d in style_rules(CSS)
        if ":not([open])" in s or 'data-expanded="false"' in s
        or 'aria-expanded="false"' in s or "[inert]" in s
    ]
    assert collapsed
    assert any("visibility: hidden" in d and "pointer-events: none" in d
               for _, d in collapsed)
    for selector, body in collapsed:
        if "opacity: 0" in body:  # the old bug: invisible but still focusable
            assert "visibility: hidden" in body, selector
            assert "pointer-events: none" in body, selector
    # Only native <details> gets disclosure styling; a div with a <summary>
    # child was never a semantic disclosure.
    summaries = [s for s, _ in style_rules(CSS) if "summary" in s and "::after" in s]
    assert summaries and all("details" in s for s in summaries)
    assert not re.search(r'(?m)^\s*(?:\.disclosure|\[data-motion="disclosure"\]) >', CSS)


def test_modal_focus_is_not_hidden_behind_an_opacity_animation() -> None:
    modal = [(s, d) for s, d in style_rules(CSS) if s.strip().startswith(".modal,")]
    assert modal and "animation: none" in modal[0][1]
    backdrop = [(s, d) for s, d in style_rules(CSS)
                if s.strip().startswith(".modal-backdrop:not(")]
    assert backdrop and "motion-view-in" in backdrop[0][1]


def test_status_flash_is_bounded_and_never_delays_state_color() -> None:
    assert "transition: none" in rule(CSS, ".status, .badge, .chat-turn-state, .connection")
    dot = rule(CSS, '.status__dot, [data-motion="status-dot"]')
    assert "background: currentColor" in dot and "transition: none" in dot
    flashes = [
        d for s, d in style_rules(CSS)
        if "::after" in s and "status-dot" in s and "animation:" in d
    ]
    assert flashes
    for body in flashes:
        assert "motion-dot-flash" in body
        assert "var(--motion-one-iteration)" in body
        assert "both" in body
    assert '[data-changing="true"]::after' in CSS
    flash_keyframes = keyframe_bodies(CSS)["motion-dot-flash"]
    assert re.search(r"100%\s*\{[^}]*opacity:\s*0", flash_keyframes)


def test_drawer_rules_are_hidden_attribute_aware() -> None:
    hidden = [(s, d) for s, d in style_rules(CSS) if ".drawer[hidden]" in s]
    assert hidden
    for _, body in hidden:
        assert "opacity: 0" in body and "visibility: hidden" in body
        assert "translate3d(var(--motion-shift-panel)" in body
    opened = [(s, d) for s, d in style_rules(CSS) if ".drawer:not([hidden])" in s]
    assert opened and "transform: none" in opened[0][1]
    assert "display var(--dur-panel) allow-discrete" in rule(CSS, '.drawer, [data-motion="drawer"]')
    starting = at_rule(CSS, "@starting-style")
    assert ".drawer:not([hidden])" in starting and "opacity: 0" in starting
    # The shipped Agent Activity drawer really does toggle [hidden] in JS.
    assert re.search(r'class="drawer"[^>]*\bhidden\b', AGENTS_LIVE_HTML)

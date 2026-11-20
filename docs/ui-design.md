# LEO UI Design Brief — "Night Instrument"

**Status:** design source of truth for the dashboard and Electron shell.
**Product:** LEO — a local, provider-independent autonomous AI agent OS.
**Audience:** technical operators and power users running agents on their own
machine; they need dense state, precise affordances and zero theatre.
**Feel:** calm, instrument-grade, premium. Never neon-gamer, never generic SaaS.

The earlier UI was a dark sidebar plus raw JSON panels. It exposed state but
gave no hierarchy, no tactile feedback and no sense of control. This system
keeps the density an operator needs while giving every screen a clear primary
action and a quiet, legible rhythm.

## 1. Visual direction

**Concept: Night Instrument.** A control surface borrowed from laboratory
measurement gear and nautical navigation instruments: matte graphite panels,
hairline etched separators, warm ivory type, a single signal-teal accent used
only for live state and the primary action, and amber/rose reserved for
attention and danger. A faint 24px measurement grid sits behind content at 2%
opacity — felt, not seen.

Three reference qualities:

1. **Oscilloscope restraint** — information-dense but rigorously aligned; every
   number has a fixed-width column and every panel has a baseline.
2. **Aerospace switchgear** — controls look operable: pressed, armed, latched
   and disabled states are visibly different; feedback is immediate but quiet.
3. **Museum wayfinding** — typography does the heavy lifting; large display
   headings, generous leading, no competing decoration.

Anti-reference: glowing green terminal clichés, purple gradients, glassmorphism
cards floating over gradients, emoji icons, centered hero layouts.

## 2. Design tokens

All values below are implemented as CSS custom properties in
`leo/interfaces/static/ui/theme.css` (primitive → semantic → component layers).
No component may hard-code a color, radius, duration or spacing value.

### Color primitives and semantic palette

| Token | Value | Use |
|---|---|---|
| `--bg-base` | `#0A0E12` | app canvas |
| `--bg-sunken` | `#070A0D` | code wells, recessed areas |
| `--surface-1` | `#11171D` | panels, sidebar |
| `--surface-2` | `#172028` | raised cards, inputs |
| `--surface-3` | `#1D2832` | hover, active rows |
| `--border-subtle` | `#24313C` | hairlines between sections |
| `--border-strong` | `#344653` | control borders, dividers |
| `--text-primary` | `#E9EFEA` | headings, primary values |
| `--text-secondary` | `#B3C0C6` | body copy |
| `--text-muted` | `#8A9AA3` | labels, metadata (AA on `--surface-1`) |
| `--accent` | `#58D6C1` | primary action, live signal, focus |
| `--accent-soft` | `rgba(88, 214, 193, .14)` | focus wash, selected row |
| `--on-accent` | `#051512` | text on accent |
| `--success` | `#7FD49B` | healthy, completed |
| `--warn` | `#E8B86D` | pending, degraded |
| `--danger` | `#F28B82` | failed, destructive |
| `--info` | `#8FB8E8` | informational links |

Status colors never carry meaning alone: every status also has a label and a
shape (dot, ring, triangle, cross) so color-blind users get the same signal.

### Scales

- **Spacing:** `--space-1: 4px`, `--space-2: 8px`, `--space-3: 12px`,
  `--space-4: 16px`, `--space-5: 24px`, `--space-6: 32px`, `--space-7: 48px`.
- **Radius:** `--radius-xs: 4px`, `--radius-sm: 8px`, `--radius-md: 12px`,
  `--radius-lg: 16px`, `--radius-pill: 999px`.
- **Shadow/glow:** `--shadow-1: 0 1px 0 rgba(255,255,255,.03), 0 2px 8px rgba(0,0,0,.24)`;
  `--shadow-2: 0 12px 32px rgba(0,0,0,.42)`;
  `--glow-accent: 0 0 0 1px var(--accent-soft), 0 0 24px rgba(88,214,193,.12)`;
  `--focus-ring: 0 0 0 2px var(--bg-base), 0 0 0 4px var(--accent)`.
- **Type:** `--text-2xs: 11px`, `--text-xs: 12px`, `--text-sm: 14px`,
  `--text-base: 16px`, `--text-lg: 20px`, `--text-xl: 28px`, `--text-2xl: 36px`.
  Body is 14px minimum; line-height 1.55; headings 1.15.
- **Z-index:** `--z-base: 0`, `--z-raised: 10`, `--z-sticky: 20`,
  `--z-dropdown: 30`, `--z-modal: 40`, `--z-toast: 50`, `--z-tooltip: 60`.
- **Motion:** `--dur-fast: 150ms`, `--dur-base: 200ms`, `--dur-slow: 300ms`;
  `--ease-standard: cubic-bezier(.2,.8,.2,1)`,
  `--ease-enter: cubic-bezier(.16,1,.3,1)`,
  `--ease-exit: cubic-bezier(.4,0,1,1)`.

## 3. Typography

- **Display:** **Space Grotesk** (500/600/700) — section titles, metrics, logo.
  Its clipped terminals give an engineered, non-default character.
- **UI:** **IBM Plex Sans** (400/500/600/700) — body, labels, controls.
  Open, highly legible at 14px on dark surfaces.
- **Numeric:** **IBM Plex Mono** (400/500/600) — tokens, counts, IDs, timings,
  table numbers, so columns align and values feel measured.

Fonts are bundled locally under `leo/interfaces/static/fonts/` with OFL
licenses. The app makes **no network request at runtime**; no font CDN, no
external CSS, no remote icons.

## 4. Components inventory

Implemented once in `ui/components.css` and `ui/app.js`:

| Component | Variants / states |
|---|---|
| Button | primary, secondary, ghost, danger; hover, active, focus-visible, disabled, loading |
| Icon button | same states; `aria-label` mandatory; 36×36 minimum |
| Input / textarea / select | rest, hover, focus, filled, invalid, disabled, helper text |
| Switch / checkbox | off, on, disabled, focus-visible |
| Segmented tabs | inactive, active, keyboard arrow navigation |
| Dropdown / menu | closed, open, item hover, item active, outside-click close |
| Card / panel | default, raised, selected, empty, loading, error |
| Stat tile | label, value, delta, trend marker |
| Table | header, striped rows, hover row, status cell, empty state |
| Chat message | user, assistant, tool, system; streaming state |
| Approval card | pending, approved, denied; approve/deny actions |
| Tooltip | top/bottom, delay 300ms, Escape to dismiss |
| Modal | overlay, dialog, focus trap, Escape close |
| Toast | info/success/warn/danger, auto-dismiss, manual close |
| Progress bar / spinner | determinate, indeterminate, reduced-motion static |
| Empty / loading / error | illustrated inline SVG, explanatory text, recovery action |
| Badge / status dot | neutral, live, success, warn, danger |
| Code / JSON view | collapsible disclosure, copy action, mono font |

Every interactive element has a visible `:focus-visible` ring, a pointer cursor,
and transitions of 150–300ms. No emoji icons: all icons are inline SVG using a
single 1.6px-stroke, rounded-cap geometric style.

## 5. Layout rules

- **Frame:** 248px left rail (logo, primary navigation, status footer) + topbar
  (view title, context, health pill, one global action) + scrollable content.
- **Grid:** 12-column grid, max content width 1560px, gutters 24px. Panels span
  4/6/8/12 columns; chat uses 8 + 4 detail rail.
- **Hierarchy:** one `h1` per view, one primary action per view. Secondary
  actions sit in the topbar or panel headers. Raw JSON is hidden behind a
  "Data" disclosure by default.
- **Default visible:** health, current objective, active tasks, pending
  approvals, recent events. Hidden until requested: full JSON, token input,
  advanced evaluation, audit table.
- **Responsive:** ≥1440px full rail and multi-column panels; 1280×720 keeps the
  rail but compresses rows and stats; <1100px rail collapses to 72px icon rail;
  <800px rail becomes a horizontal scrollable tab bar, panels stack, tables
  become cards, no horizontal page scroll.
- **Electron shell:** same dashboard at `127.0.0.1`; no separate visual
  language. Window chrome stays native.

## 6. Motion rules

- Panel entrance: opacity 0→1, translateY 8px→0, 200ms `--ease-enter`, 40ms
  stagger for stat tiles only.
- Button press: background/shadow change 150ms, translateY 1px on active.
- Menu/modal: scale .98→1 + opacity, 150ms `--ease-enter`; exit 120ms
  `--ease-exit`.
- Toast: slide from bottom 16px, 200ms; auto-dismiss after 5s.
- Live status: a 2px dot pulses 2s ease-in-out; it stops when data is stale.
- No animation of width/height or layout-affecting properties.
- `@media (prefers-reduced-motion: reduce)` removes transforms, pulses and
  stagger; only opacity fades of 100ms remain, and live indicators show a
  static "LIVE" label instead of pulsing.

"""Email channel adapter — SMTP sending plus IMAP polling.

Email is LEO's first poll-based messaging channel: there is no signed
webhook ingress, so inbound mail is *pulled* from an injectable IMAP
transport while outbound mail goes through an injectable SMTP transport.
The default transports wrap :mod:`smtplib` and :mod:`imaplib` (stdlib
only) inside :func:`asyncio.to_thread`; tests inject fakes and never
touch a socket — this adapter makes no HTTP calls at all.

Configuration is split into two halves:

* Send half (``credential_env``): ``SMTP_HOST``, ``SMTP_USER``,
  ``SMTP_PASSWORD`` and ``EMAIL_FROM`` (``EMAIL_FROM`` is an address,
  not a secret, but it stays in ``credential_env`` so ``record()``
  reflects the full required outbound configuration). ``SMTP_PORT`` is
  an optional override, default 587.
* Poll half (``poll_env``): ``IMAP_HOST``, ``IMAP_USER``,
  ``IMAP_PASSWORD`` (plus optional ``IMAP_PORT``, default 993). These
  are deliberately NOT part of ``credential_env``: the framework's
  generic ``record()`` would then report the *send* half as missing
  whenever only IMAP is. Instead ``record()`` is overridden to merge
  ``poll_env`` into ``config_ref`` and the metadata, so a record
  honestly reflects both halves and is "configured" only when both
  exist.

The poll cursor is a set of seen UIDs held on the adapter instance —
per-process memory, not durable state. A restarted process re-reads
whatever the server still lists as UNSEEN. The default IMAP transport
fetches with ``BODY.PEEK`` and never flags, expunges or deletes
anything, so the mailbox is left untouched. UIDs are stable per mailbox
but are not validated against ``UIDVALIDITY`` changes (a mailbox
recreated under the same name resets them).

Credentials are read from the environment by *name* at call time; no
password value is ever stored on the instance, logged, or echoed in an
error message. Inbound mail is untrusted data: every field is
type-checked and coerced, bodies are capped, and nothing in a message
is ever treated as an instruction.

Transport security: SMTP uses implicit TLS on 465 and *verified* STARTTLS
(``ssl.create_default_context()``) on other ports; a server that does not
offer STARTTLS is rejected before any login. IMAP uses implicit TLS on 993
and verified STARTTLS elsewhere. Credentials are never sent over a cleartext
socket.

The ``From`` header is not an identity. A sender is only surfaced as its
address when ``Authentication-Results`` (SPF/DKIM, domain-aligned) or
``Received-SPF`` evidence passes; otherwise the sender is degraded to
``unverified:<address>`` so exact-match allowlists deny it by default.
"""

from __future__ import annotations

import asyncio
import imaplib
import re
import smtplib
import ssl
from datetime import timezone
from email import message_from_bytes, policy
from email.mime.text import MIMEText
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Callable, Protocol

import httpx

from ...contracts import IntegrationRecord, utcnow
from ..messaging import HeaderMap, MessagingAdapter, MessagingError, NormalizedMessage
from ..secrets import credential

#: Inbound bodies are capped at this many characters during normalization:
#: email is untrusted data and a single poll must never flood the gateway.
MAX_BODY_CHARS = 10_000
#: Defensive upper bound on messages the default IMAP transport fetches
#: per poll so one poll cannot run unbounded work.
MAX_FETCH_PER_POLL = 50


def _env_int(name: str, default: int) -> int:
    """Optional integer override via the credential broker; never raises."""
    raw = credential(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _as_text(value: Any) -> str:
    """Coerce an untrusted transport field to plain text without raising."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _email_date(value: str) -> str:
    """RFC 2822 date -> ISO-8601 timestamp; ``utcnow()`` when unparseable."""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return utcnow()
    if parsed is None or parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc).isoformat() if parsed else utcnow()
    return parsed.isoformat()


def _validated_address(target: Any) -> str:
    """Basic recipient validation: exactly one ``@``, non-empty on both
    sides, no newlines (a header-injection guard). Raises MessagingError."""
    address = _as_text(target).strip()
    if address.count("@") != 1 or "\n" in address or "\r" in address:
        raise MessagingError("email send blocked: invalid email address")
    local, _, domain = address.partition("@")
    if not local or not domain:
        raise MessagingError("email send blocked: invalid email address")
    return address


def _auth_domain(value: str) -> str:
    """Domain carried by an SPF/DKIM parameter value; ``""`` when unusable."""
    value = value.strip().strip("<>").split()[0] if value.strip() else ""
    if not value:
        return ""
    domain = value.rpartition("@")[2].strip().lower().rstrip(".")
    return domain if domain and "." in domain else ""


def _sender_auth_verified(address: str, item: Any) -> bool:
    """Whether the untrusted ``From`` address has aligned SPF/DKIM evidence.

    ``Authentication-Results`` clauses and ``Received-SPF`` values are data,
    never instructions. A pass is accepted only when the authenticated domain
    equals (or is a subdomain of) the ``From`` domain. Failures, absent
    headers and malformed values all return False.
    """
    domain = _auth_domain(address)
    if not domain:
        return False
    values: list[str] = []
    if isinstance(item, dict):
        for key in ("authentication_results", "Authentication-Results", "received_spf", "Received-SPF"):
            raw = item.get(key)
            pieces = [raw] if isinstance(raw, str) else (
                [value for value in raw if isinstance(value, str)] if isinstance(raw, (list, tuple)) else []
            )
            for value in pieces:
                values.append(f"received-spf {value}" if "received" in key.lower() else value)
    text = " ".join(values).lower()
    if not text:
        return False
    markers = {
        "spf": ("smtp.mailfrom=", "envelope-from="),
        "dkim": ("header.from=", "header.d="),
    }
    for clause in text.replace("\n", " ").split(";"):
        mechanism = next((name for name in markers if f"{name}=pass" in clause), None)
        if mechanism is None:
            continue
        for marker in markers[mechanism]:
            position = clause.find(marker)
            if position < 0:
                continue
            tail = clause[position + len(marker) :]
            found = _auth_domain(tail.split()[0] if tail.split() else "")
            if found and (found == domain or found.endswith("." + domain)):
                return True
    # Received-SPF: pass (...) ...; envelope-from=user@example.com; helo=...
    if re.search(r"\breceived-spf\s*:?\s*pass\b", text):
        match = re.search(r"envelope-from=([^\s;]+)", text)
        if match and _auth_domain(match.group(1)) == domain:
            return True
    return False


class EmailSmtpTransport(Protocol):
    """Outbound email seam. Implementations must not leak credentials in
    exceptions; the adapter only ever surfaces the exception *type name*."""

    def send_message(self, from_addr: str, to_addr: str, subject: str, text: str) -> None: ...


class EmailImapTransport(Protocol):
    """Inbound email seam: the currently-UNSEEN messages as plain dicts
    with the keys ``uid``, ``from``, ``subject``, ``body`` and ``date``."""

    def fetch_unseen(self) -> list[dict]: ...


class _SmtplibTransport:
    """Default SMTP transport over stdlib :mod:`smtplib`, run in a worker thread.

    Credentials are read from the environment at call time. Port 465 uses
    implicit TLS; every other port must advertise STARTTLS. TLS is verified
    with ``ssl.create_default_context()`` and login never happens on a
    cleartext connection.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def send_message(self, from_addr: str, to_addr: str, subject: str, text: str) -> None:
        host = credential("SMTP_HOST")
        user = credential("SMTP_USER")
        password = credential("SMTP_PASSWORD")
        if not host or not user or not password:
            missing = next(
                name
                for name, value in (("SMTP_HOST", host), ("SMTP_USER", user), ("SMTP_PASSWORD", password))
                if not value
            )
            raise MessagingError(f"email send blocked: {missing} is not set")
        message = MIMEText(str(text))  # text is passed through verbatim
        message["Subject"] = str(subject)
        message["From"] = str(from_addr)
        message["To"] = str(to_addr)
        port = _env_int("SMTP_PORT", 587)
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(str(host), port, timeout=self.timeout, context=context) as smtp:
                smtp.ehlo_or_helo_if_needed()
                smtp.login(str(user), str(password))
                smtp.send_message(message)
            return
        with smtplib.SMTP(str(host), port, timeout=self.timeout) as smtp:
            smtp.ehlo_or_helo_if_needed()
            if not smtp.has_extn("starttls"):
                # Refuse before login: smtplib would otherwise send AUTH on a
                # cleartext socket.
                raise MessagingError(
                    "email send blocked: SMTP server does not offer STARTTLS; refusing plaintext login"
                )
            smtp.starttls(context=context)
            smtp.ehlo()
            smtp.login(str(user), str(password))
            smtp.send_message(message)


def _parse_message(uid: bytes, raw: bytes) -> dict:
    """Best-effort extraction of one raw RFC 2822 message; never raises."""
    record: dict[str, Any] = {
        "uid": uid.decode("ascii", "replace"),
        "from": "",
        "subject": "",
        "body": "",
        "date": "",
        "authentication_results": "",
        "received_spf": "",
    }
    try:
        parsed = message_from_bytes(raw, policy=policy.default)
    except Exception:  # noqa: BLE001 - an unparseable message yields empty fields
        return record
    for target, header in (
        ("from", "From"),
        ("subject", "Subject"),
        ("date", "Date"),
        ("authentication_results", "Authentication-Results"),
        ("received_spf", "Received-SPF"),
    ):
        try:
            record[target] = str(parsed.get(header) or "")
        except Exception:  # noqa: BLE001 - defective headers must not kill the poll
            continue
    try:
        body_part = parsed.get_body(preferencelist=("plain",))
        if body_part is not None:
            content = body_part.get_content()
            if isinstance(content, str):
                record["body"] = content
    except Exception:  # noqa: BLE001
        pass
    return record


class _ImaplibTransport:
    """Default IMAP transport over stdlib :mod:`imaplib`, run in a worker thread.

    Reads ``IMAP_HOST``/``IMAP_USER``/``IMAP_PASSWORD`` at call time,
    selects INBOX, searches for UNSEEN and fetches at most
    ``MAX_FETCH_PER_POLL`` bodies with ``BODY.PEEK`` (non-destructive:
    nothing is flagged, expunged or deleted). Malformed messages are
    skipped, never fatal.

    Port 993 uses implicit TLS; every other port must support STARTTLS.
    ``ssl.create_default_context()`` verifies the certificate and hostname,
    and login is never attempted before TLS is established.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def fetch_unseen(self) -> list[dict]:
        host = credential("IMAP_HOST")
        user = credential("IMAP_USER")
        password = credential("IMAP_PASSWORD")
        if not host or not user or not password:
            missing = next(
                name
                for name, value in (("IMAP_HOST", host), ("IMAP_USER", user), ("IMAP_PASSWORD", password))
                if not value
            )
            raise MessagingError(f"email poll blocked: {missing} is not set")
        messages: list[dict] = []
        port = _env_int("IMAP_PORT", 993)
        context = ssl.create_default_context()
        if port == 993:
            connection: imaplib.IMAP4 = imaplib.IMAP4_SSL(
                str(host), port, timeout=self.timeout, ssl_context=context
            )
        else:
            connection = imaplib.IMAP4(str(host), port, timeout=self.timeout)
        with connection as imap:
            if port != 993:
                # starttls() raises IMAP4.error when the server lacks CAPABILITY
                # STARTTLS; no login follows, so credentials stay off the wire.
                imap.starttls(ssl_context=context)
            imap.login(str(user), str(password))
            status, _data = imap.select("INBOX")
            if status != "OK":
                return messages
            status, data = imap.uid("SEARCH", None, "UNSEEN")
            if status != "OK" or not data or not data[0]:
                return messages
            for uid in data[0].split()[:MAX_FETCH_PER_POLL]:
                item = self._fetch_one(imap, uid)
                if item is not None:
                    messages.append(item)
        return messages

    def _fetch_one(self, imap: imaplib.IMAP4, uid: bytes) -> dict | None:
        try:
            status, parts = imap.uid("FETCH", uid, "(BODY.PEEK[])")
            if status != "OK" or not parts:
                return None
            raw = next(
                (
                    bytes(part[1])
                    for part in parts
                    if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], (bytes, bytearray))
                ),
                None,
            )
            if raw is None:
                return None
            return _parse_message(uid, raw)
        except (imaplib.IMAP4.error, ValueError, TypeError):
            return None


class EmailAdapter(MessagingAdapter):
    """SMTP send + IMAP poll adapter. Email has no webhook ingress at all:
    ``supports_webhook`` is False and ``verify_webhook`` is a hard ``False``
    so no unsigned inbound path can ever be routed here."""

    channel = "email"
    credential_env = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_FROM")
    verification_env = ()
    #: IMAP half of the configuration. Kept separate from credential_env so
    #: the framework's generic record() cannot mistake "IMAP missing" for
    #: "sending broken"; record() below merges it back in.
    poll_env = ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD")
    supports_webhook = False
    supports_polling = True

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        require_verification: bool = True,
        signature_verifier: "Callable[[str, str, str, bytes], bool] | None" = None,
        smtp_transport: "EmailSmtpTransport | None" = None,
        imap_transport: "EmailImapTransport | None" = None,
    ) -> None:
        super().__init__(
            client=client,
            timeout=timeout,
            require_verification=require_verification,
            signature_verifier=signature_verifier,
        )
        # Injectable transports: tests (and alternative mail backends) never
        # touch a real socket. The defaults only connect inside a worker
        # thread when actually used.
        self.smtp_transport: EmailSmtpTransport = smtp_transport or _SmtplibTransport()
        self.imap_transport: EmailImapTransport = imap_transport or _ImaplibTransport()
        # Poll cursor: UIDs already returned. Per-process memory, NOT durable
        # state — see the module docstring for the honest consequences.
        self.seen_uids: set[str] = set()

    def verify_webhook(self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None) -> bool:
        # Email has no signed webhook ingress; polling is the only inbound path.
        return False

    async def handle_webhook(
        self, payload: Any, headers: HeaderMap = None, *, raw_body: bytes | None = None
    ) -> list[NormalizedMessage]:
        raise MessagingError("email does not support webhook ingress; use poll()")

    async def send(self, target: str, text: str) -> dict[str, Any]:
        from_addr = credential("EMAIL_FROM")
        missing = next(
            (
                name
                for name, value in (
                    ("SMTP_HOST", credential("SMTP_HOST")),
                    ("SMTP_USER", credential("SMTP_USER")),
                    ("SMTP_PASSWORD", credential("SMTP_PASSWORD")),
                    ("EMAIL_FROM", from_addr),
                )
                if not value
            ),
            None,
        )
        if missing:
            raise MessagingError(f"email send blocked: {missing} is not set")
        to_addr = _validated_address(target)
        try:
            await asyncio.to_thread(
                self.smtp_transport.send_message, from_addr, to_addr, "LEO message", str(text)
            )
        except MessagingError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface the type name only, never the text
            raise MessagingError(f"email send failed: {type(exc).__name__}") from None
        return {"channel": self.channel, "target": to_addr, "success": True, "message_id": None}

    async def poll(self) -> list[NormalizedMessage]:
        """Pull UNSEEN mail through the IMAP transport and normalize it.

        The seen-UID cursor is instance state: a second poll never returns
        the same messages, but a new adapter (or a restarted process)
        starts empty and re-reads whatever the server still lists as
        UNSEEN. Transport failures surface as ``email poll failed:
        <ExceptionTypeName>`` — never the exception text, which may
        contain hostnames or credentials.
        """
        try:
            fetched = await asyncio.to_thread(self.imap_transport.fetch_unseen)
        except MessagingError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface the type name only
            raise MessagingError(f"email poll failed: {type(exc).__name__}") from None
        messages: list[NormalizedMessage] = []
        if not isinstance(fetched, list):
            return messages
        for item in fetched:
            # UNTRUSTED DATA: every field is type-checked and coerced;
            # malformed items are skipped, never fatal.
            if not isinstance(item, dict):
                continue
            uid = item.get("uid")
            if isinstance(uid, bool):
                continue
            uid_key = uid if isinstance(uid, str) else str(uid) if isinstance(uid, int) else None
            if not uid_key or not uid_key.strip() or uid_key in self.seen_uids:
                continue
            _, address = parseaddr(_as_text(item.get("from")))
            # The From header is attacker-controlled. Only an aligned
            # SPF/DKIM pass makes it usable as an admission identity;
            # everything else is degraded so exact-match allowlists deny it.
            if not address:
                sender = "unverified"
            elif _sender_auth_verified(address, item):
                sender = address
            else:
                sender = f"unverified:{address}"
            subject = _as_text(item.get("subject"))
            body = _as_text(item.get("body"))[:MAX_BODY_CHARS]
            messages.append(
                NormalizedMessage(
                    channel=self.channel,
                    sender=sender,
                    text=subject + "\n\n" + body,
                    timestamp=_email_date(_as_text(item.get("date"))),
                    reply_ref=None,
                    message_id=uid_key,
                )
            )
            self.seen_uids.add(uid_key)
        return messages

    def _is_configured(self) -> bool:
        # Honest reporting: fully configured only when BOTH the SMTP (send)
        # and the IMAP (poll) halves exist. The base implementation only
        # knows credential_env, i.e. the send half.
        return all(credential(name) for name in self.credential_env + self.poll_env)

    def record(self) -> IntegrationRecord:
        """Merge the IMAP half into the framework record.

        The base record only reflects ``credential_env`` + ``verification_env``
        (email has no verification_env), so without this override the record
        would hide the poll configuration entirely. Status stays "configured"
        only when both halves exist; a missing half is named in the reason.
        """
        record = super().record()
        record.config_ref = ",".join(self.credential_env + self.poll_env)
        record.metadata["poll_env"] = list(self.poll_env)
        record.metadata["poll_credentials_present"] = [name for name in self.poll_env if credential(name)]
        missing_send = [name for name in self.credential_env if not credential(name)]
        missing_poll = [name for name in self.poll_env if not credential(name)]
        if record.status == "configured":
            record.metadata["reason"] = None
        else:
            parts = []
            if missing_send:
                parts.append(f"missing send credentials: {', '.join(missing_send)}")
            if missing_poll:
                parts.append(f"missing poll credentials: {', '.join(missing_poll)}")
            record.metadata["reason"] = "; ".join(parts)
        return record


__all__ = ["EmailAdapter", "EmailImapTransport", "EmailSmtpTransport"]

"""Opt-in WhatsApp Web device-link bridge (library boundary).

Nothing here is imported or started by the LEO app automatically. Enabling it
is an explicit operator/maintainer decision with ToS and account-ban tradeoffs;
see README.md.
"""

from .manager import (
    BridgeAuthError,
    BridgeConfig,
    BridgeError,
    BridgeNotEnabled,
    BridgeTimeout,
    BridgeUnavailable,
    SessionAuthError,
    SessionOwnershipError,
    WhatsAppBridgeManager,
)

__all__ = [
    "BridgeAuthError",
    "BridgeConfig",
    "BridgeError",
    "BridgeNotEnabled",
    "BridgeTimeout",
    "BridgeUnavailable",
    "SessionAuthError",
    "SessionOwnershipError",
    "WhatsAppBridgeManager",
]

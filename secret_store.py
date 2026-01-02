"""
Secret storage helper for MaiVecMem plugin.

Provides cross-platform storage of a randomly generated AES-256 key using the OS keyring
(if available) and local-file fallback. Exposes simple encrypt/decrypt helpers using AES-GCM.

Usage:
    from .secret_store import encrypt_for_service, decrypt_for_service

Notes:
- Requires `keyring` (preferred) and `cryptography` packages. If not available, will fall back
  to a local `.secret_key` file in the plugin directory (with restricted permissions when possible).
- The encryption output is base64(nonce || ciphertext). Nonce length is 12 bytes (recommended for AES-GCM).
"""
from __future__ import annotations

import os
import base64

PLUGIN_DIR = os.path.dirname(__file__)
LOCAL_KEY_FILE = os.path.join(PLUGIN_DIR, ".secret_key")
_KEYRING_SERVICE = "MaiVecMem_secret_service"
_KEYRING_USER = "MaiVecMem_aes_key"

try:
    import keyring
except Exception:
    keyring = None

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:
    AESGCM = None


def _ensure_key_bytes() -> bytes | None:
    """Return 32 bytes AES key, creating/storing it if necessary.

    Storage preference: system keyring (if available) -> local file fallback.
    """
    # Try keyring first
    if keyring is not None:
        try:
            val = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
            if val:
                return base64.b64decode(val)
            # create key and save
            k = os.urandom(32)
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, base64.b64encode(k).decode())
            return k
        except Exception:
            # fallback to file
            pass

    # File fallback
    try:
        if os.path.exists(LOCAL_KEY_FILE):
            with open(LOCAL_KEY_FILE, "rb") as f:
                data = f.read().strip()
                return base64.b64decode(data)
        else:
            k = os.urandom(32)
            b64 = base64.b64encode(k)
            # write with restrictive permissions when possible
            try:
                with open(LOCAL_KEY_FILE, "wb") as f:
                    f.write(b64)
                try:
                    os.chmod(LOCAL_KEY_FILE, 0o600)
                except Exception:
                    pass
            except Exception:
                return None
            return k
    except Exception:
        return None


def encrypt_for_service(plaintext: str) -> str:
    """Encrypt plaintext string and return base64-encoded nonce||ciphertext."""
    if AESGCM is None:
        raise RuntimeError("cryptography package is required for encryption")
    key = _ensure_key_bytes()
    if not key:
        raise RuntimeError("Failed to obtain encryption key")
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode()


def decrypt_for_service(token_b64: str) -> str:
    """Decrypt base64-encoded nonce||ciphertext and return plaintext string."""
    if AESGCM is None:
        raise RuntimeError("cryptography package is required for decryption")
    key = _ensure_key_bytes()
    if not key:
        raise RuntimeError("Failed to obtain encryption key")
    try:
        raw = base64.b64decode(token_b64)
        nonce = raw[:12]
        ct = raw[12:]
        aes = AESGCM(key)
        pt = aes.decrypt(nonce, ct, None)
        return pt.decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Decryption failed: {e}") from e


def key_available() -> bool:
    """Return True if encryption/decryption is possible in this environment."""
    return AESGCM is not None and _ensure_key_bytes() is not None


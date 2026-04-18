from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Callable, Mapping, Optional

from bilibili_api import Credential
from bilibili_api.exceptions import (
    CookiesRefreshException,
    CredentialNoAcTimeValueException,
    CredentialNoBiliJctException,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

CONFIG_SECTION = "credential"
CONFIG_ASR_SECTION = "asr"


class CredentialStoreError(RuntimeError):
    pass


@dataclass(slots=True)
class StoredCredential:
    sessdata: Optional[str] = None
    bili_jct: Optional[str] = None
    buvid3: Optional[str] = None
    buvid4: Optional[str] = None
    dedeuserid: Optional[str] = None
    ac_time_value: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "StoredCredential":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            sessdata=_clean_str(data.get("sessdata")),
            bili_jct=_clean_str(data.get("bili_jct")),
            buvid3=_clean_str(data.get("buvid3")),
            buvid4=_clean_str(data.get("buvid4")),
            dedeuserid=_clean_str(data.get("dedeuserid")),
            ac_time_value=_clean_str(data.get("ac_time_value")),
            updated_at=_clean_str(data.get("updated_at")),
        )

    @classmethod
    def from_cookie_string(cls, cookie_header: str) -> "StoredCredential":
        cookie = SimpleCookie()
        cookie.load(cookie_header)
        return cls(
            sessdata=_cookie_value(cookie, "SESSDATA"),
            bili_jct=_cookie_value(cookie, "bili_jct"),
            buvid3=_cookie_value(cookie, "buvid3"),
            buvid4=_cookie_value(cookie, "buvid4"),
            dedeuserid=_cookie_value(cookie, "DedeUserID"),
            ac_time_value=_cookie_value(cookie, "ac_time_value"),
        )

    @classmethod
    def from_credential(cls, credential: Credential) -> "StoredCredential":
        return cls(
            sessdata=credential.sessdata,
            bili_jct=credential.bili_jct,
            buvid3=credential.buvid3,
            buvid4=credential.buvid4,
            dedeuserid=credential.dedeuserid,
            ac_time_value=credential.ac_time_value,
            updated_at=_utc_now_iso(),
        )

    def merged_with(self, fallback: "StoredCredential") -> "StoredCredential":
        return StoredCredential(
            sessdata=self.sessdata or fallback.sessdata,
            bili_jct=self.bili_jct or fallback.bili_jct,
            buvid3=self.buvid3 or fallback.buvid3,
            buvid4=self.buvid4 or fallback.buvid4,
            dedeuserid=self.dedeuserid or fallback.dedeuserid,
            ac_time_value=self.ac_time_value or fallback.ac_time_value,
            updated_at=self.updated_at or fallback.updated_at,
        )

    def to_credential(self) -> Credential:
        return Credential(
            sessdata=self.sessdata,
            bili_jct=self.bili_jct,
            buvid3=self.buvid3,
            buvid4=self.buvid4,
            dedeuserid=self.dedeuserid,
            ac_time_value=self.ac_time_value,
        )

    def to_mapping(self) -> dict[str, str]:
        data = {
            "sessdata": self.sessdata,
            "bili_jct": self.bili_jct,
            "buvid3": self.buvid3,
            "buvid4": self.buvid4,
            "dedeuserid": self.dedeuserid,
            "ac_time_value": self.ac_time_value,
            "updated_at": self.updated_at,
        }
        return {key: value for key, value in data.items() if value}

    def is_empty(self) -> bool:
        return not any(
            [
                self.sessdata,
                self.bili_jct,
                self.buvid3,
                self.buvid4,
                self.dedeuserid,
                self.ac_time_value,
            ]
        )

    def supports_refresh(self) -> bool:
        return bool(self.sessdata and self.bili_jct and self.ac_time_value)

    def missing_refresh_fields(self) -> list[str]:
        missing = []
        if not self.sessdata:
            missing.append("SESSDATA")
        if not self.bili_jct:
            missing.append("bili_jct")
        if not self.ac_time_value:
            missing.append("ac_time_value")
        return missing


def get_default_config_path() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "bilibili-subtitle-fetch" / "config.toml"

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "bilibili-subtitle-fetch" / "config.toml"

    return Path.home() / ".config" / "bilibili-subtitle-fetch" / "config.toml"


def load_stored_credential(config_path: Path) -> StoredCredential:
    if not config_path.exists():
        return StoredCredential()

    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised via manager
        raise CredentialStoreError(
            f"Failed to parse credential config: {config_path}"
        ) from exc

    return StoredCredential.from_mapping(data.get(CONFIG_SECTION))


def load_asr_config(config_path: Path) -> dict[str, object]:
    """Load the [asr] section from config.toml, returning a plain dict."""
    if not config_path.exists():
        return {}
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    section = data.get(CONFIG_ASR_SECTION)
    if not isinstance(section, dict):
        return {}
    return section


def save_stored_credential(config_path: Path, stored: StoredCredential) -> None:
    stored.updated_at = _utc_now_iso()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content_lines = [f"[{CONFIG_SECTION}]"]
    for key, value in stored.to_mapping().items():
        content_lines.append(f'{key} = "{_escape_toml(value)}"')
    config_path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")

    if os.name != "nt":
        config_path.chmod(0o600)


def initialize_credential_file(
    config_path: Path,
    prompt: Optional[Callable[[str], str]] = None,
    output: Optional[Callable[[str], None]] = None,
) -> Path:
    prompt = prompt or input
    output = output or print

    existing = load_stored_credential(config_path)
    output(f"Credential file: {config_path}")
    output(
        "Paste the full Bilibili Cookie header. "
        "Leave it empty to input fields one by one."
    )
    cookie_header = prompt("Cookie: ").strip()

    if cookie_header:
        entered = StoredCredential.from_cookie_string(cookie_header)
    else:
        entered = StoredCredential()

    stored = entered.merged_with(existing)

    if not stored.sessdata:
        stored.sessdata = _prompt_secret(prompt, "SESSDATA (required): ")
    if not stored.bili_jct:
        stored.bili_jct = _prompt_secret(
            prompt, "bili_jct (required for auto refresh): "
        )
    if not stored.ac_time_value:
        stored.ac_time_value = _prompt_secret(
            prompt, "ac_time_value (required for auto refresh): "
        )
    if not stored.buvid3:
        stored.buvid3 = _prompt_secret(prompt, "buvid3 (optional): ")
    if not stored.buvid4:
        stored.buvid4 = _prompt_secret(prompt, "buvid4 (optional): ")
    if not stored.dedeuserid:
        stored.dedeuserid = _prompt_secret(prompt, "DedeUserID (optional): ")

    if not stored.sessdata:
        raise CredentialStoreError(
            "SESSDATA is required to initialize the credential file."
        )

    save_stored_credential(config_path, stored)
    output(f"Saved Bilibili credentials to {config_path}")
    if stored.supports_refresh():
        output("Automatic cookie refresh is enabled.")
    else:
        missing = ", ".join(stored.missing_refresh_fields())
        output(
            "Automatic cookie refresh is disabled until these fields are provided: "
            f"{missing}"
        )
    return config_path


class CredentialManager:
    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self._config_path = (
            Path(config_path).expanduser() if config_path else get_default_config_path()
        )
        self._has_custom_config_path = config_path is not None
        self._lock: Optional[asyncio.Lock] = None
        self._stored = StoredCredential()
        self._credential = Credential()
        self._loaded = False

    @property
    def config_path(self) -> Path:
        return self._config_path

    def set_config_path(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path).expanduser()
        self._has_custom_config_path = True
        self._loaded = False
        self._stored = StoredCredential()
        self._credential = Credential()

    def load(self) -> StoredCredential:
        self._stored = load_stored_credential(self._config_path)
        self._credential = self._stored.to_credential()
        self._loaded = True
        return self._stored

    def validate_runtime_config(self) -> StoredCredential:
        if not self._config_path.exists():
            raise CredentialStoreError(
                "Credential config not found. "
                f"Run `{self._get_init_command()}` first."
            )

        stored = self.load()
        if not stored.sessdata:
            raise CredentialStoreError(
                "Credential config is missing `sessdata`. "
                f"Update {self._config_path} or run `bilibili-subtitle-fetch init` again."
            )

        return stored

    def _get_init_command(self) -> str:
        if not self._has_custom_config_path:
            return "bilibili-subtitle-fetch init"

        return f'bilibili-subtitle-fetch init --config "{self._config_path}"'

    async def get_credential(self) -> tuple[Credential, Optional[str]]:
        lock = self._lock or asyncio.Lock()
        self._lock = lock

        async with lock:
            if not self._loaded:
                self.validate_runtime_config()

            if not self._stored.supports_refresh():
                return self._credential, None

            try:
                needs_refresh = await self._credential.check_refresh()
            except Exception as exc:
                return (
                    self._credential,
                    f"Automatic cookie refresh check failed: {exc}",
                )

            if not needs_refresh:
                return self._credential, None

            try:
                await self._credential.refresh()
            except (
                CookiesRefreshException,
                CredentialNoAcTimeValueException,
                CredentialNoBiliJctException,
            ) as exc:
                return self._credential, f"Automatic cookie refresh failed: {exc}"
            except Exception as exc:
                return self._credential, f"Automatic cookie refresh failed: {exc}"

            self._stored = StoredCredential.from_credential(self._credential)
            try:
                save_stored_credential(self._config_path, self._stored)
            except Exception as exc:
                return (
                    self._credential,
                    "Automatic cookie refresh succeeded but saving the updated config "
                    f"failed: {exc}",
                )

            return (
                self._credential,
                f"Automatically refreshed Bilibili cookies and saved {self._config_path}.",
            )


def _cookie_value(cookie: SimpleCookie[str], key: str) -> Optional[str]:
    morsel = cookie.get(key)
    if morsel is None:
        return None
    return morsel.value or None


def _clean_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _escape_toml(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _prompt_secret(prompt: Callable[[str], str], label: str) -> Optional[str]:
    value = prompt(label).strip()
    return value or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

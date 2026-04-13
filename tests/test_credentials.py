import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bilibili_api import Credential

from bilibili_subtitle_fetch.credentials import (
    CredentialStoreError,
    CredentialManager,
    StoredCredential,
    initialize_credential_file,
    load_stored_credential,
    save_stored_credential,
)


class CredentialHelpersTest(unittest.TestCase):
    def test_from_cookie_string_extracts_refresh_fields(self) -> None:
        stored = StoredCredential.from_cookie_string(
            "SESSDATA=sess; bili_jct=csrf; ac_time_value=refresh; "
            "DedeUserID=42; buvid3=buvid3; buvid4=buvid4"
        )

        self.assertEqual(stored.sessdata, "sess")
        self.assertEqual(stored.bili_jct, "csrf")
        self.assertEqual(stored.ac_time_value, "refresh")
        self.assertEqual(stored.dedeuserid, "42")
        self.assertEqual(stored.buvid3, "buvid3")
        self.assertEqual(stored.buvid4, "buvid4")

    def test_save_and_load_roundtrip(self) -> None:
        stored = StoredCredential(
            sessdata="sess",
            bili_jct="csrf",
            buvid3="buvid3",
            buvid4="buvid4",
            dedeuserid="42",
            ac_time_value="refresh",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            save_stored_credential(config_path, stored)
            loaded = load_stored_credential(config_path)

        self.assertEqual(loaded.sessdata, "sess")
        self.assertEqual(loaded.bili_jct, "csrf")
        self.assertEqual(loaded.buvid3, "buvid3")
        self.assertEqual(loaded.buvid4, "buvid4")
        self.assertEqual(loaded.dedeuserid, "42")
        self.assertEqual(loaded.ac_time_value, "refresh")
        self.assertIsNotNone(loaded.updated_at)

    def test_initialize_credential_file_accepts_cookie_header(self) -> None:
        prompts = iter(
            [
                "SESSDATA=sess; bili_jct=csrf; ac_time_value=refresh; "
                "DedeUserID=42; buvid3=buvid3; buvid4=buvid4"
            ]
        )
        outputs: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            initialize_credential_file(
                config_path,
                prompt=lambda _: next(prompts),
                output=outputs.append,
            )
            loaded = load_stored_credential(config_path)

        self.assertEqual(loaded.sessdata, "sess")
        self.assertEqual(loaded.ac_time_value, "refresh")
        self.assertIn("Automatic cookie refresh is enabled.", outputs)


class CredentialManagerTest(unittest.TestCase):
    def test_validate_runtime_config_requires_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            manager = CredentialManager()
            manager._config_path = config_path

            with self.assertRaises(CredentialStoreError) as ctx:
                manager.validate_runtime_config()

        self.assertIn("bilibili-subtitle-fetch init", str(ctx.exception))
        self.assertNotIn("--config", str(ctx.exception))

    def test_validate_runtime_config_uses_config_flag_for_custom_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "custom.toml"
            manager = CredentialManager(config_path)

            with self.assertRaises(CredentialStoreError) as ctx:
                manager.validate_runtime_config()

        self.assertIn(f'--config "{config_path}"', str(ctx.exception))

    def test_manager_auto_refreshes_and_persists_updated_cookie(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            save_stored_credential(
                config_path,
                StoredCredential(
                    sessdata="sess",
                    bili_jct="csrf",
                    ac_time_value="refresh",
                    dedeuserid="1",
                ),
            )
            manager = CredentialManager(config_path)

            async def fake_check_refresh(self: Credential) -> bool:
                return True

            async def fake_refresh(self: Credential) -> None:
                self.sessdata = "sess-new"
                self.bili_jct = "csrf-new"
                self.ac_time_value = "refresh-new"
                self.dedeuserid = "2"

            with patch.object(Credential, "check_refresh", fake_check_refresh):
                with patch.object(Credential, "refresh", fake_refresh):
                    credential, note = asyncio.run(manager.get_credential())

            saved = load_stored_credential(config_path)

        self.assertEqual(credential.sessdata, "sess-new")
        self.assertEqual(saved.sessdata, "sess-new")
        self.assertEqual(saved.bili_jct, "csrf-new")
        self.assertEqual(saved.ac_time_value, "refresh-new")
        self.assertEqual(saved.dedeuserid, "2")
        self.assertIn("Automatically refreshed Bilibili cookies", note)


if __name__ == "__main__":
    unittest.main()

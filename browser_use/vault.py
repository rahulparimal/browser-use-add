"""Local encrypted vault for confidential values the agent may type into pages.

The LLM only ever sees placeholders (`<secret>name</secret>`); `Agent(sensitive_data=...)` substitutes the real
value locally at action time, and only on pages whose domain matches the entry. This vault persists those values:

	<config>/vault.db   SQLite; values are Fernet-encrypted (AES-128-CBC + HMAC-SHA256)
	<config>/vault.key  Fernet key, mode 0600 (or set BU_VAULT_KEY to supply your own)

Threat model: protects values at rest against casual reads/backups/sync of the DB file. It does NOT protect
against malware running as your user (it can read the key file too). Prefer BU_VAULT_KEY from a keychain for more.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from cryptography.fernet import Fernet, InvalidToken
from pydantic import AfterValidator, BaseModel, ConfigDict


def _non_blank(v: str) -> str:
	assert v.strip(), 'must not be blank'
	return v.strip()


NonBlank = Annotated[str, AfterValidator(_non_blank)]


class VaultEntry(BaseModel):
	"""Metadata only: values never leave `Vault.sensitive_data()`."""

	model_config = ConfigDict(frozen=True)

	domain: str
	name: str
	updated_at: datetime


def default_vault_dir() -> Path:
	from browser_use.config import CONFIG

	return Path(os.environ.get('BU_VAULT_DIR') or CONFIG.BROWSER_USE_CONFIG_DIR)


class Vault:
	def __init__(self, directory: str | Path | None = None):
		self.directory = Path(directory) if directory else default_vault_dir()
		self.db_path = self.directory / 'vault.db'
		self.key_path = self.directory / 'vault.key'
		self._fernet: Fernet | None = None

	def _key(self) -> bytes:
		env = os.environ.get('BU_VAULT_KEY')
		if env:
			return env.encode()
		if self.key_path.exists():
			return self.key_path.read_bytes().strip()
		self.directory.mkdir(parents=True, exist_ok=True)
		key = Fernet.generate_key()
		# O_EXCL + 0600: never expose the key even briefly, and never clobber an existing one
		fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
		with os.fdopen(fd, 'wb') as f:
			f.write(key)
		return key

	@property
	def fernet(self) -> Fernet:
		if self._fernet is None:
			self._fernet = Fernet(self._key())
		return self._fernet

	def _connect(self) -> sqlite3.Connection:
		self.directory.mkdir(parents=True, exist_ok=True)
		new = not self.db_path.exists()
		conn = sqlite3.connect(self.db_path)
		if new:
			os.chmod(self.db_path, 0o600)
		conn.execute(
			'CREATE TABLE IF NOT EXISTS secrets ('
			'domain TEXT NOT NULL, name TEXT NOT NULL, ciphertext BLOB NOT NULL, updated_at TEXT NOT NULL, '
			'PRIMARY KEY (domain, name))'
		)
		return conn

	def set(self, domain: str, name: str, value: str) -> None:
		domain, name = _non_blank(domain), _non_blank(name)
		assert value, 'value must not be empty'
		token = self.fernet.encrypt(value.encode())
		with closing(self._connect()) as conn, conn:
			conn.execute(
				'INSERT INTO secrets (domain, name, ciphertext, updated_at) VALUES (?, ?, ?, ?) '
				'ON CONFLICT(domain, name) DO UPDATE SET ciphertext=excluded.ciphertext, updated_at=excluded.updated_at',
				(domain, name, token, datetime.now(timezone.utc).isoformat()),
			)

	def delete(self, domain: str, name: str) -> bool:
		with closing(self._connect()) as conn, conn:
			return conn.execute('DELETE FROM secrets WHERE domain=? AND name=?', (domain, name)).rowcount > 0

	def list(self) -> list[VaultEntry]:
		with closing(self._connect()) as conn:
			rows = conn.execute('SELECT domain, name, updated_at FROM secrets ORDER BY domain, name').fetchall()
		return [VaultEntry(domain=d, name=n, updated_at=datetime.fromisoformat(u)) for d, n, u in rows]

	def sensitive_data(self, domains: list[str] | None = None) -> dict[str, str | dict[str, str]]:
		"""`{domain: {name: value}}`, the domain-scoped format `Agent(sensitive_data=...)` expects."""
		out: dict[str, dict[str, str]] = {}  # inner dicts only; widened on return for Agent's invariant type
		with closing(self._connect()) as conn:
			rows = conn.execute('SELECT domain, name, ciphertext FROM secrets').fetchall()
		for domain, name, token in rows:
			if domains is not None and domain not in domains:
				continue
			try:
				out.setdefault(domain, {})[name] = self.fernet.decrypt(token).decode()
			except InvalidToken:
				raise ValueError(f'Cannot decrypt {domain}/{name}: wrong vault key (BU_VAULT_KEY or {self.key_path})') from None
		return {k: v for k, v in out.items()}

	def domains(self) -> list[str]:
		return sorted({e.domain for e in self.list()})

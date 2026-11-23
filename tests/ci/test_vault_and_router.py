"""Vault (encrypted SQLite) and privacy router (local vs cloud model switching)."""

import os
import stat

import pytest
from pydantic import BaseModel

from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage
from browser_use.llm.router import PrivacyRouterChat, detect_sensitive
from browser_use.llm.views import ChatInvokeCompletion
from browser_use.vault import Vault


# region - vault
def test_roundtrip_and_encrypted_at_rest(tmp_path):
	vault = Vault(tmp_path)
	vault.set('example.com', 'password', 'hunter2-very-secret')
	assert vault.sensitive_data() == {'example.com': {'password': 'hunter2-very-secret'}}
	assert b'hunter2-very-secret' not in (tmp_path / 'vault.db').read_bytes()


def test_files_are_owner_only(tmp_path):
	Vault(tmp_path).set('a.com', 'k', 'v')
	for name in ('vault.db', 'vault.key'):
		assert stat.S_IMODE(os.stat(tmp_path / name).st_mode) == 0o600


def test_list_never_exposes_values_and_update_delete(tmp_path):
	vault = Vault(tmp_path)
	vault.set('a.com', 'user', 'alice')
	vault.set('a.com', 'user', 'alice2')  # upsert
	vault.set('b.com', 'user', 'bob')
	entries = vault.list()
	assert [(e.domain, e.name) for e in entries] == [('a.com', 'user'), ('b.com', 'user')]
	assert 'alice' not in repr(entries)
	assert vault.sensitive_data(domains=['a.com']) == {'a.com': {'user': 'alice2'}}
	assert vault.delete('b.com', 'user') and not vault.delete('b.com', 'user')
	assert vault.domains() == ['a.com']


def test_wrong_key_fails_loudly(tmp_path, monkeypatch):
	Vault(tmp_path).set('a.com', 'k', 'v')
	monkeypatch.setenv('BU_VAULT_KEY', 'ZmFrZWtleWZha2VrZXlmYWtla2V5ZmFrZWtleWZha2U=')
	with pytest.raises(ValueError, match='wrong vault key'):
		Vault(tmp_path).sensitive_data()


def test_rejects_blank_inputs(tmp_path):
	vault = Vault(tmp_path)
	with pytest.raises(AssertionError):
		vault.set('a.com', '  ', 'v')
	with pytest.raises(AssertionError):
		vault.set('a.com', 'k', '')


# endregion


# region - router
class _Reply(BaseModel):
	who: str


class _LLM:
	"""Real object implementing the chat protocol; records what it was sent (the allowed LLM stand-in)."""

	_verified_api_keys = True

	def __init__(self, who: str, fail: bool = False):
		self.who, self.fail, self.model, self.seen = who, fail, f'{who}-model', []

	provider = property(lambda self: self.who)
	name = property(lambda self: self.who)

	async def ainvoke(self, messages, output_format=None, **kwargs):
		self.seen.append(messages)
		if self.fail:
			raise ConnectionError('ollama down')
		return ChatInvokeCompletion(completion=_Reply(who=self.who), usage=None)


def _state(url: str, body: str = '') -> list:
	text = f'<agent_history></agent_history><browser_state>\nCurrent URL: {url}\n{body}\n</browser_state>'
	img = ContentPartImageParam(image_url=ImageURL(url='data:image/png;base64,AAAA'))
	return [SystemMessage(content='sys'), UserMessage(content=[ContentPartTextParam(text=text), img])]


def test_detect_sensitive():
	assert detect_sensitive('<browser_state>https://bank.example.com/x</browser_state>', ['https://*.example.com'])
	assert detect_sensitive('card 4111 1111 1111 1111', [])  # Luhn-valid test card
	assert detect_sensitive('ssn 123-45-6789', [])
	assert detect_sensitive('mail me@x.io', []) and not detect_sensitive('mail me@x.io', [], detect_email=False)
	assert detect_sensitive('order 1234 5678 9012 3456', []) is None  # fails Luhn
	assert detect_sensitive('plain news page, 84 points', ['bank.com']) is None
	# vault domain only counts inside <browser_state>, not in e.g. the task text
	assert (
		detect_sensitive(
			'<user_request>go to bank.com</user_request><browser_state>news.ycombinator.com</browser_state>', ['bank.com']
		)
		is None
	)


async def test_routes_per_step_and_strips_images_locally():
	cloud, local = _LLM('cloud'), _LLM('local')
	router = PrivacyRouterChat(cloud, local, sensitive_domains=['bank.example.com'])

	assert (await router.ainvoke(_state('https://news.ycombinator.com'), _Reply)).completion.who == 'cloud'
	assert (await router.ainvoke(_state('https://bank.example.com/login'), _Reply)).completion.who == 'local'
	assert router.calls == {'cloud': 1, 'local': 1}
	assert any(isinstance(p, ContentPartImageParam) for p in cloud.seen[0][1].content)
	assert not any(isinstance(p, ContentPartImageParam) for p in local.seen[0][1].content)
	assert router.model == 'cloud-model'


async def test_fails_closed_unless_opted_in():
	router = PrivacyRouterChat(_LLM('cloud'), _LLM('local', fail=True), sensitive_domains=['bank.com'])
	with pytest.raises(RuntimeError, match='refusing to send it to the cloud'):
		await router.ainvoke(_state('https://bank.com'), _Reply)
	assert router.calls['cloud'] == 0

	lenient = PrivacyRouterChat(_LLM('cloud'), _LLM('local', fail=True), sensitive_domains=['bank.com'], fallback_to_cloud=True)
	assert (await lenient.ainvoke(_state('https://bank.com'), _Reply)).completion.who == 'cloud'


# endregion

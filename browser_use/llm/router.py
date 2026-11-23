"""Privacy router: send steps that expose sensitive page content to a local model, everything else to the cloud.

Routing is decided per call from the current browser state text:
  * the page URL/host matches a domain you stored in the vault, or
  * the page text contains PII (email, Luhn-valid card number, SSN-shaped number).
Sensitive steps go to `local` with screenshots stripped (a screenshot would leak the same content, and many local
models are text-only). If the local model fails the router FAILS CLOSED: it does not retry the sensitive step on the
cloud unless `fallback_to_cloud=True`.

Limits: detection is heuristic. A page showing private data in a form the patterns miss (names, addresses, balances)
still reaches the cloud model unless its domain is in `sensitive_domains`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage, ContentPartImageParam, UserMessage
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_CARD_RE = re.compile(r'\b(?:\d[ -]?){13,19}\b')
_STATE_RE = re.compile(r'<browser_state>(.*?)</browser_state>', re.DOTALL)


def _luhn_ok(digits: str) -> bool:
	total, alt = 0, False
	for ch in reversed(digits):
		d = int(ch)
		if alt:
			d = d * 2 - 9 if d > 4 else d * 2
		total += d
		alt = not alt
	return total % 10 == 0


def _host_of(pattern: str) -> str:
	return pattern.split('://')[-1].removeprefix('*.').split('/')[0].lower()


def detect_sensitive(text: str, sensitive_domains: list[str], detect_email: bool = True) -> str | None:
	"""Return a short reason if `text` looks sensitive, else None (never includes the matched value)."""
	state = _STATE_RE.search(text)
	scope = (state.group(1) if state else text).lower()
	for pattern in sensitive_domains:
		host = _host_of(pattern)
		if host and host in scope:
			return f'vault domain {host}'
	if _SSN_RE.search(scope):
		return 'SSN-shaped number on page'
	for m in _CARD_RE.finditer(scope):
		digits = re.sub(r'\D', '', m.group())
		if 13 <= len(digits) <= 19 and _luhn_ok(digits):
			return 'card number on page'
	if detect_email and _EMAIL_RE.search(scope):
		return 'email address on page'
	return None


def _strip_images(messages: list[BaseMessage]) -> list[BaseMessage]:
	out: list[BaseMessage] = []
	for m in messages:
		if isinstance(m, UserMessage) and isinstance(m.content, list):
			kept = [p for p in m.content if not isinstance(p, ContentPartImageParam)]
			out.append(m.model_copy(update={'content': kept}))
		else:
			out.append(m)
	return out


class PrivacyRouterChat:
	"""Drop-in `BaseChatModel` that delegates to `cloud` or `local` per call."""

	_verified_api_keys: bool = True

	def __init__(
		self,
		cloud: BaseChatModel,
		local: BaseChatModel,
		sensitive_domains: list[str] | None = None,
		fallback_to_cloud: bool = False,
		detect_email: bool = True,
	):
		self.cloud = cloud
		self.local = local
		self.sensitive_domains = sensitive_domains or []
		self.fallback_to_cloud = fallback_to_cloud
		self.detect_email = detect_email
		self.model = cloud.model
		self.calls = {'cloud': 0, 'local': 0}
		self._last_route: str | None = None

	@property
	def provider(self) -> str:
		return self.cloud.provider

	@property
	def name(self) -> str:
		return f'router({self.cloud.name}|{self.local.name})'

	@property
	def model_name(self) -> str:
		return self.model

	def _state_text(self, messages: list[BaseMessage]) -> str:
		for m in reversed(messages):
			if isinstance(m, UserMessage):
				return m.text
		return ''

	def _log_route(self, route: str, reason: str | None) -> None:
		if route != self._last_route:
			logger.info(f'🔀 LLM route -> {route}' + (f' ({reason})' if reason else ''))
			self._last_route = route

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		reason = detect_sensitive(self._state_text(messages), self.sensitive_domains, self.detect_email)
		if reason is None:
			self._log_route('cloud', None)
			self.calls['cloud'] += 1
			return await self.cloud.ainvoke(messages, output_format, **kwargs)

		self._log_route('local', reason)
		self.calls['local'] += 1
		try:
			return await self.local.ainvoke(_strip_images(messages), output_format, **kwargs)
		except Exception as e:
			if not self.fallback_to_cloud:
				raise RuntimeError(
					f'Local model failed on a sensitive step ({reason}); refusing to send it to the cloud. '
					f'Fix the local model or pass fallback_to_cloud=True. Cause: {e}'
				) from e
			logger.warning(f'Local model failed ({e}); falling back to cloud as configured')
			self.calls['cloud'] += 1
			return await self.cloud.ainvoke(messages, output_format, **kwargs)

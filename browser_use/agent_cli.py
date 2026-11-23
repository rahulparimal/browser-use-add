"""`bu-agent`: run browser tasks and manage replayable skills from the terminal.

	bu-agent run "find the cheapest laptop on example.com" --save-skill laptop-price
	bu-agent skills list | find "laptop" | run laptop-price --set email=a@b.c | delete laptop-price
	bu-agent vault set example.com password            # prompts for the value; stored encrypted in SQLite
	bu-agent run "log in to example.com" --use-vault --local-model ollama:llama3.1
	bu-agent calibrate --model openai:gpt-6-luna         # measure real image token cost for this model

(`browser-use` / `bu` is the separate Browser Harness CLI.)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel

# (env var, provider, default model) in auto-detection order
_PROVIDERS = [
	('ANTHROPIC_API_KEY', 'anthropic', 'claude-sonnet-4-6'),
	('OPENAI_API_KEY', 'openai', 'gpt-4o'),
	('GOOGLE_API_KEY', 'google', 'gemini-2.5-flash'),
]
_LOCAL_PROVIDERS = {'ollama': 'llama3.1'}  # no API key; needs a running Ollama server


def _openai_rejects_sampling_params(model: str) -> bool:
	m = re.match(r'(?:gpt-(\d+)|o(\d+))', model.lower())
	return bool(m) and (m.group(2) is not None or int(m.group(1)) >= 5)


def resolve_llm(model: str | None) -> BaseChatModel:
	"""`provider:model`, bare `provider`, or auto-detect from whichever API key is set."""
	provider, _, name = (model or '').partition(':')
	if not provider:
		for env, prov, _default in _PROVIDERS:
			if os.environ.get(env):
				provider = prov
				break
		else:
			raise SystemExit('No LLM configured: set ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY, or pass --model.')
	default = {p: m for _, p, m in _PROVIDERS}.get(provider) or _LOCAL_PROVIDERS.get(provider)
	if default is None:
		raise SystemExit(f'Unknown provider {provider!r}; use one of {[p for _, p, _ in _PROVIDERS] + list(_LOCAL_PROVIDERS)}')
	name = name or default
	if provider == 'ollama':
		from browser_use.llm.ollama.chat import ChatOllama

		return ChatOllama(model=name)
	if provider == 'anthropic':
		from browser_use.llm.anthropic.chat import ChatAnthropic

		return ChatAnthropic(model=name)
	if provider == 'openai':
		from browser_use.llm.openai.chat import ChatOpenAI

		# GPT-5+ and o-series models only accept the default sampling settings (a 400 otherwise)
		if _openai_rejects_sampling_params(name):
			return ChatOpenAI(model=name, temperature=None, frequency_penalty=None)
		return ChatOpenAI(model=name)
	from browser_use.llm.google.chat import ChatGoogle

	return ChatGoogle(model=name)


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(prog='bu-agent', description='Token-efficient browser agent with replayable skills.')
	sub = p.add_subparsers(dest='cmd', required=True)

	run = sub.add_parser('run', help='run a task with the agent')
	run.add_argument('task')
	run.add_argument('--model', help='provider[:model], e.g. anthropic:claude-sonnet-4-6 (default: auto from env keys)')
	run.add_argument('--headless', action='store_true')
	run.add_argument('--max-steps', type=int, default=30)
	run.add_argument('--save-skill', metavar='NAME', help='record a successful run as a replayable skill')
	run.add_argument('--no-optimize-images', action='store_true', help='send raw screenshots')
	run.add_argument('--skip-unchanged', action='store_true', help='omit screenshots identical to the previous step')
	run.add_argument('--no-vision', action='store_true', help='same as --vision off')
	run.add_argument(
		'--vision',
		choices=['on', 'off', 'auto'],
		default=None,
		help='on: screenshot every step (default); off: DOM text only; auto: screenshot only when the agent asks for one',
	)
	run.add_argument(
		'--use-vault', action='store_true', help='fill stored confidential values via placeholders (never sent to the LLM)'
	)
	run.add_argument(
		'--local-model', metavar='ollama:MODEL', help='route steps that expose sensitive page content to this local model'
	)
	run.add_argument(
		'--local-fallback-cloud',
		action='store_true',
		help='if the local model fails, send the step to the cloud anyway (off = fail closed)',
	)

	vault = sub.add_parser('vault', help='manage confidential values (encrypted SQLite)')
	vsub = vault.add_subparsers(dest='vault_cmd', required=True)
	vset = vsub.add_parser('set', help='store a value (prompted, never taken from argv)')
	vset.add_argument('domain', help='where it may be used, e.g. example.com or https://*.example.com')
	vset.add_argument('name', help='placeholder name the LLM will see, e.g. password')
	vset.add_argument('--from-env', metavar='VAR', help='read the value from this environment variable instead of prompting')
	vsub.add_parser('list', help='show domains and names (never values)')
	vdel = vsub.add_parser('delete')
	vdel.add_argument('domain')
	vdel.add_argument('name')

	cal = sub.add_parser('calibrate', help='measure real image token cost for a model (3 tiny API calls)')
	cal.add_argument('--model', help='provider:model to measure')

	skills = sub.add_parser('skills', help='manage local skills')
	ssub = skills.add_subparsers(dest='skills_cmd', required=True)
	ssub.add_parser('list')
	find = ssub.add_parser('find')
	find.add_argument('query')
	show = ssub.add_parser('show')
	show.add_argument('name')
	delete = ssub.add_parser('delete')
	delete.add_argument('name')
	replay = ssub.add_parser('run', help='replay a skill without an LLM-driven loop')
	replay.add_argument('name')
	replay.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='parameter value (repeatable)')
	replay.add_argument('--model', help='LLM used only for fallback steps and the summary')
	replay.add_argument('--headless', action='store_true')
	return p


def _parse_sets(pairs: list[str]) -> dict[str, str]:
	out: dict[str, str] = {}
	for pair in pairs:
		key, sep, value = pair.partition('=')
		if not sep or not key:
			raise SystemExit(f'--set expects KEY=VALUE, got {pair!r}')
		out[key] = value
	return out


def _log_usage(summary) -> None:
	print(
		f'tokens: prompt={summary.total_prompt_tokens} (cached={summary.total_prompt_cached_tokens}) '
		f'completion={summary.total_completion_tokens} cost=${summary.total_cost:.4f}',
		file=sys.stderr,
	)


def _resolve_vision(args: argparse.Namespace) -> bool | Literal['auto']:
	if args.vision is not None and args.no_vision:
		raise SystemExit('Use either --vision or --no-vision, not both.')
	if args.no_vision:
		return False
	return {None: True, 'on': True, 'off': False, 'auto': 'auto'}[args.vision]


async def _run_task(args: argparse.Namespace) -> int:
	from browser_use import Agent, Browser
	from browser_use.screenshots.optimizer import ImageOptimizerSettings
	from browser_use.vault import Vault

	use_vision = _resolve_vision(args)
	sensitive_data: dict[str, str | dict[str, str]] | None = None
	if args.use_vault:
		sensitive_data = Vault().sensitive_data()
		if not sensitive_data:
			raise SystemExit('Vault is empty: add values with `bu-agent vault set DOMAIN NAME`.')

	llm = resolve_llm(args.model)
	router = None
	if args.local_model:
		from browser_use.llm.router import PrivacyRouterChat

		router = PrivacyRouterChat(
			cloud=llm,
			local=resolve_llm(args.local_model),
			sensitive_domains=list(sensitive_data or {}),
			fallback_to_cloud=args.local_fallback_cloud,
		)
		llm = cast('BaseChatModel', router)  # structurally a chat model; the protocol's overloads defeat inference
	optimizer = (
		False
		if args.no_optimize_images
		else ImageOptimizerSettings.for_model(getattr(llm, 'model', ''), skip_unchanged=args.skip_unchanged)
	)
	agent = Agent(
		task=args.task,
		llm=llm,
		browser=Browser(headless=args.headless),
		use_vision=use_vision,
		sensitive_data=sensitive_data,
		image_optimizer=optimizer,
		record_skill=args.save_skill,
		calculate_cost=True,
	)
	history = await agent.run(max_steps=args.max_steps)
	print(history.final_result() or '(no result)')
	if router is not None:
		print(f'routing: cloud={router.calls["cloud"]} local={router.calls["local"]} calls', file=sys.stderr)
	try:
		_log_usage(await agent.token_cost_service.get_usage_summary())
	except Exception:
		pass
	return 0 if history.is_successful() else 1


async def _replay(args: argparse.Namespace) -> int:
	from browser_use import Browser
	from browser_use.skills.local import LocalSkillStore

	store = LocalSkillStore()
	results = await store.run(
		args.name, llm=resolve_llm(args.model), variables=_parse_sets(args.set), browser_session=Browser(headless=args.headless)
	)
	last = results[-1] if results else None
	print((last.extracted_content if last else None) or '(no result)')
	return 0 if last is not None and not last.error else 1


def _skills_command(args: argparse.Namespace) -> int:
	from browser_use.skills.local import LocalSkillStore

	store = LocalSkillStore()
	if args.skills_cmd == 'list':
		for s in store.list():
			print(f'{s.name}\t{s.description}\t(params: {", ".join(s.parameters) or "-"}; used {s.use_count}x)')
	elif args.skills_cmd == 'find':
		for s, score in store.find(args.query):
			print(f'{score:.2f}\t{s.name}\t{s.description}')
	elif args.skills_cmd == 'show':
		try:
			print((store.root / args.name / 'SKILL.md').read_text(encoding='utf-8'))
		except FileNotFoundError:
			raise SystemExit(f'Skill {args.name!r} not found in {store.root}')
	elif args.skills_cmd == 'delete':
		try:
			store.delete(args.name)
		except FileNotFoundError:
			raise SystemExit(f'Skill {args.name!r} not found in {store.root}')
	return 0


def _vault_command(args: argparse.Namespace) -> int:
	import getpass

	from browser_use.vault import Vault

	vault = Vault()
	if args.vault_cmd == 'set':
		value = os.environ.get(args.from_env, '') if args.from_env else getpass.getpass(f'Value for {args.domain}/{args.name}: ')
		if not value:
			raise SystemExit('Empty value; nothing stored.')
		vault.set(args.domain, args.name, value)
		print(f'stored {args.domain}/{args.name}')
	elif args.vault_cmd == 'list':
		for e in vault.list():
			print(f'{e.domain}\t{e.name}\t(updated {e.updated_at:%Y-%m-%d %H:%M})')
	elif args.vault_cmd == 'delete':
		if not vault.delete(args.domain, args.name):
			raise SystemExit(f'{args.domain}/{args.name} not found')
	return 0


async def _calibrate(args: argparse.Namespace) -> int:
	from browser_use.screenshots.calibration import calibrate_image_tokens, save_calibration

	llm = resolve_llm(args.model)
	cal = await calibrate_image_tokens(llm)
	path = save_calibration(cal)
	print(f'{cal.model}: ~{cal.base_tokens:.0f} + {cal.tokens_per_pixel * 1e6:.1f} tokens per megapixel (saved to {path})')
	for w, h in [(768, 576), (1024, 768), (1280, 800), (1920, 1080)]:
		print(f'  {w}x{h}: ~{cal.estimate(w, h)} tokens')
	return 0


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	if args.cmd == 'run':
		return asyncio.run(_run_task(args))
	if args.cmd == 'vault':
		return _vault_command(args)
	if args.cmd == 'calibrate':
		return asyncio.run(_calibrate(args))
	if args.skills_cmd == 'run':
		return asyncio.run(_replay(args))
	return _skills_command(args)


if __name__ == '__main__':
	sys.exit(main())

"""`bu-agent`: run browser tasks and manage replayable skills from the terminal.

	bu-agent run "find the cheapest laptop on example.com" --save-skill laptop-price
	bu-agent skills list | find "laptop" | run laptop-price --set email=a@b.c | delete laptop-price

(`browser-use` / `bu` is the separate Browser Harness CLI.)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel

# (env var, provider, default model) in auto-detection order
_PROVIDERS = [
	('ANTHROPIC_API_KEY', 'anthropic', 'claude-sonnet-4-6'),
	('OPENAI_API_KEY', 'openai', 'gpt-4o'),
	('GOOGLE_API_KEY', 'google', 'gemini-2.5-flash'),
]


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
	default = {p: m for _, p, m in _PROVIDERS}.get(provider)
	if default is None:
		raise SystemExit(f'Unknown provider {provider!r}; use one of {[p for _, p, _ in _PROVIDERS]}')
	name = name or default
	if provider == 'anthropic':
		from browser_use.llm.anthropic.chat import ChatAnthropic

		return ChatAnthropic(model=name)
	if provider == 'openai':
		from browser_use.llm.openai.chat import ChatOpenAI

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
	run.add_argument('--no-vision', action='store_true', help='DOM text only (cheapest)')

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


async def _run_task(args: argparse.Namespace) -> int:
	from browser_use import Agent, Browser
	from browser_use.screenshots.optimizer import ImageOptimizerSettings

	llm = resolve_llm(args.model)
	optimizer = (
		False
		if args.no_optimize_images
		else ImageOptimizerSettings.for_model(getattr(llm, 'model', ''), skip_unchanged=args.skip_unchanged)
	)
	agent = Agent(
		task=args.task,
		llm=llm,
		browser=Browser(headless=args.headless),
		use_vision=not args.no_vision,
		image_optimizer=optimizer,
		record_skill=args.save_skill,
		calculate_cost=True,
	)
	history = await agent.run(max_steps=args.max_steps)
	print(history.final_result() or '(no result)')
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


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	if args.cmd == 'run':
		return asyncio.run(_run_task(args))
	if args.skills_cmd == 'run':
		return asyncio.run(_replay(args))
	return _skills_command(args)


if __name__ == '__main__':
	sys.exit(main())

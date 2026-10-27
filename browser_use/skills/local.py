"""Local, offline skills: record a successful agent run once, replay it later without an LLM.

A skill is a directory:

	<root>/<name>/skill.json    metadata, task and parameter schema (pydantic `LocalSkill`)
	<root>/<name>/history.json  the recorded `AgentHistoryList` (sensitive data already redacted)
	<root>/<name>/SKILL.md      human/Claude-readable summary of the steps

Replay re-resolves elements against the live DOM (`Agent.rerun_history`) and only calls the LLM for
fallback AI steps and the final summary, so the happy path costs ~0 tokens.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

if TYPE_CHECKING:
	from browser_use.agent.views import ActionResult, AgentHistoryList
	from browser_use.browser import BrowserSession
	from browser_use.llm.base import BaseChatModel

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{0,63}$')
_TOKEN_RE = re.compile(r'[a-z0-9]+')


def _validate_skill_name(name: str) -> str:
	# also guards against path traversal, since the name becomes a directory
	if not _NAME_RE.fullmatch(name):
		raise ValueError(f'Invalid skill name {name!r}: use 1-64 chars of [a-z0-9_-], starting with a letter or digit')
	return name


SkillName = Annotated[str, AfterValidator(_validate_skill_name)]


def default_skills_dir() -> Path:
	from browser_use.config import CONFIG

	return Path(os.environ.get('BROWSER_USE_LOCAL_SKILLS_DIR') or CONFIG.BROWSER_USE_CONFIG_DIR / 'local_skills')


class LocalSkillParameter(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	name: str
	type: str = 'string'
	format: str | None = None
	example: str | None = None  # omitted for values that were secrets in the recorded run


class LocalSkill(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	name: SkillName
	description: str
	task: str
	parameters: dict[str, LocalSkillParameter] = Field(default_factory=dict)
	steps: int = 0
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	use_count: int = 0
	last_used_at: datetime | None = None


def _tokens(text: str) -> list[str]:
	return _TOKEN_RE.findall(text.lower())


def _secret_placeholders(sensitive_data: dict[str, Any] | None) -> dict[str, str]:
	"""Map each secret value to its `<secret>key</secret>` placeholder (same form history.json uses)."""
	out: dict[str, str] = {}
	for key, v in (sensitive_data or {}).items():
		values = v.values() if isinstance(v, dict) else [v]
		for x in values:
			if x:
				out[str(x)] = f'<secret>{key}</secret>'
	return out


def _render_steps(history: AgentHistoryList, secrets: dict[str, str], max_value_chars: int = 80) -> list[str]:
	lines: list[str] = []
	for item in history.history:
		if not item.model_output:
			continue
		for action in item.model_output.action:
			dumped = action.model_dump(exclude_none=True, exclude_unset=True)
			for name, params in dumped.items():
				if name == 'interacted_element':
					continue
				args = ', '.join(f'{k}={str(v)[:max_value_chars]!r}' for k, v in (params or {}).items())
				for value, placeholder in secrets.items():
					args = args.replace(value, placeholder)
				lines.append(f'{name}({args})')
	return lines


class LocalSkillStore:
	def __init__(self, root: str | Path | None = None):
		self.root = Path(root) if root else default_skills_dir()

	def _dir(self, name: str) -> Path:
		return self.root / _validate_skill_name(name)

	def history_path(self, name: str) -> Path:
		return self._dir(name) / 'history.json'

	def save(
		self,
		name: str,
		history: AgentHistoryList,
		task: str,
		description: str | None = None,
		sensitive_data: dict[str, Any] | None = None,
		overwrite: bool = False,
	) -> LocalSkill:
		"""Persist a recorded run. Refuses unfinished/unsuccessful runs: replaying those teaches nothing."""
		from browser_use.agent.variable_detector import detect_variables_in_history

		skill_dir = self._dir(name)
		if skill_dir.exists() and not overwrite:
			raise FileExistsError(f'Skill {name!r} already exists (pass overwrite=True to replace it)')
		if history.is_successful() is not True:
			raise ValueError('Only successfully completed runs can be saved as a skill')

		secrets = _secret_placeholders(sensitive_data)
		parameters = {
			var.name: LocalSkillParameter(
				name=var.name,
				type=var.type,
				format=var.format,
				example=None if var.original_value in secrets else var.original_value,
			)
			for var in detect_variables_in_history(history).values()
		}
		steps = _render_steps(history, secrets)
		skill = LocalSkill(
			name=name,
			description=description or task.strip().splitlines()[0][:200],
			task=task,
			parameters=parameters,
			steps=len(steps),
		)

		skill_dir.mkdir(parents=True, exist_ok=True)
		history.save_to_file(skill_dir / 'history.json', sensitive_data=sensitive_data)
		(skill_dir / 'skill.json').write_text(skill.model_dump_json(indent=2), encoding='utf-8')
		(skill_dir / 'SKILL.md').write_text(self._render_skill_md(skill, steps), encoding='utf-8')
		logger.info(f'💾 Saved skill {name!r} ({len(steps)} actions, {len(parameters)} parameters) to {skill_dir}')
		return skill

	@staticmethod
	def _render_skill_md(skill: LocalSkill, steps: list[str]) -> str:
		params = '\n'.join(
			f'- `{p.name}` ({p.type}){f" e.g. `{p.example}`" if p.example else ""}' for p in skill.parameters.values()
		)
		numbered = '\n'.join(f'{i}. {s}' for i, s in enumerate(steps, 1))
		desc = skill.description.replace('\n', ' ').replace('"', "'")
		return (
			f'---\nname: {skill.name}\ndescription: "{desc}"\n---\n\n'
			f'# {skill.name}\n\n## Task\n{skill.task}\n\n'
			f'## Parameters\n{params or "_none detected_"}\n\n## Recorded actions\n{numbered}\n'
		)

	def load(self, name: str) -> LocalSkill:
		path = self._dir(name) / 'skill.json'
		if not path.exists():
			raise KeyError(f'Skill {name!r} not found in {self.root}')
		return LocalSkill.model_validate_json(path.read_text(encoding='utf-8'))

	def list(self) -> list[LocalSkill]:
		if not self.root.exists():
			return []
		skills: list[LocalSkill] = []
		for p in sorted(self.root.glob('*/skill.json')):
			try:
				skills.append(LocalSkill.model_validate_json(p.read_text(encoding='utf-8')))
			except Exception as e:
				logger.warning(f'Skipping unreadable skill {p}: {e}')
		return skills

	def delete(self, name: str) -> None:
		shutil.rmtree(self._dir(name))

	def find(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[tuple[LocalSkill, float]]:
		"""Rank skills against a free-text request (BM25-lite: IDF-weighted token overlap, name/description boosted)."""
		skills = self.list()
		q = set(_tokens(query))
		if not skills or not q:
			return []
		docs = [Counter(_tokens(f'{s.name} {s.name} {s.description} {s.description} {s.task}')) for s in skills]
		df = Counter(t for d in docs for t in d)
		n = len(docs)
		scored: list[tuple[LocalSkill, float]] = []
		for skill, doc in zip(skills, docs):
			total = sum(doc.values()) or 1
			score = sum(log(1 + n / df[t]) * (doc[t] / total) ** 0.5 for t in q if t in doc)
			if score > min_score:
				scored.append((skill, score))
		return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

	async def run(
		self,
		name: str,
		llm: BaseChatModel,
		variables: dict[str, str] | None = None,
		browser_session: BrowserSession | None = None,
		**rerun_kwargs: Any,
	) -> list[ActionResult]:
		"""Replay a skill, substituting `variables` for the recorded parameter values."""
		from browser_use import Agent

		skill = self.load(name)
		unknown = set(variables or {}) - set(skill.parameters)
		if unknown:
			raise ValueError(f'Unknown parameters {sorted(unknown)} for skill {name!r}; available: {sorted(skill.parameters)}')

		agent = Agent(task=skill.task, llm=llm, browser_session=browser_session)
		try:
			results = await agent.load_and_rerun(self.history_path(name), variables=variables, **rerun_kwargs)
		finally:
			if browser_session is None:
				await agent.close()
			skill.use_count += 1
			skill.last_used_at = datetime.now(timezone.utc)
			(self._dir(name) / 'skill.json').write_text(skill.model_dump_json(indent=2), encoding='utf-8')
		return results


__all__ = ['LocalSkill', 'LocalSkillParameter', 'LocalSkillStore', 'default_skills_dir']

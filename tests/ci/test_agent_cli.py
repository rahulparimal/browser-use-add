"""bu-agent CLI: argument parsing, LLM resolution and skill management (no browser needed)."""

import pytest

from browser_use.agent_cli import _parse_sets, build_parser, main, resolve_llm
from browser_use.skills.local import LocalSkillStore


def test_run_flags_parse():
	args = build_parser().parse_args(
		['run', 'do thing', '--headless', '--save-skill', 'x', '--skip-unchanged', '--max-steps', '5']
	)
	assert (args.task, args.headless, args.save_skill, args.skip_unchanged, args.max_steps) == ('do thing', True, 'x', True, 5)


def test_parse_sets():
	assert _parse_sets(['a=1', 'b=x=y']) == {'a': '1', 'b': 'x=y'}
	with pytest.raises(SystemExit):
		_parse_sets(['novalue'])


def test_resolve_llm_autodetects_from_env(monkeypatch):
	for k in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']:
		monkeypatch.delenv(k, raising=False)
	with pytest.raises(SystemExit, match='No LLM configured'):
		resolve_llm(None)
	monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
	assert resolve_llm(None).model == 'gpt-4o'
	monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
	assert resolve_llm(None).model == 'claude-sonnet-4-6'
	assert resolve_llm('openai:gpt-4o-mini').model == 'gpt-4o-mini'
	with pytest.raises(SystemExit, match='Unknown provider'):
		resolve_llm('nope:model')


def test_skills_list_find_show_delete(tmp_path, monkeypatch, capsys, make_history):
	monkeypatch.setenv('BROWSER_USE_LOCAL_SKILLS_DIR', str(tmp_path))
	LocalSkillStore().save('invoice-export', make_history(), task='Download invoices as PDF', description='Export invoices')

	assert main(['skills', 'list']) == 0
	assert 'invoice-export' in capsys.readouterr().out
	assert main(['skills', 'find', 'download invoices']) == 0
	assert 'invoice-export' in capsys.readouterr().out
	assert main(['skills', 'show', 'invoice-export']) == 0
	assert 'Recorded actions' in capsys.readouterr().out
	assert main(['skills', 'delete', 'invoice-export']) == 0
	assert LocalSkillStore().list() == []
	with pytest.raises(SystemExit, match='not found'):
		main(['skills', 'show', 'invoice-export'])

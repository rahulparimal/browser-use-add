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


def test_newer_openai_models_get_no_sampling_params(monkeypatch):
	monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
	for name in ['gpt-6-luna', 'gpt-5.6-luna', 'gpt-5', 'o3', 'o4-mini']:
		llm = resolve_llm(f'openai:{name}')
		assert llm.temperature is None and llm.frequency_penalty is None, name
	for name in ['gpt-4o', 'gpt-4.1-mini']:
		llm = resolve_llm(f'openai:{name}')
		assert llm.temperature == 0.2 and llm.frequency_penalty == 0.3, name


def test_vision_flag_resolution():
	from browser_use.agent_cli import _resolve_vision

	parse = build_parser().parse_args
	assert _resolve_vision(parse(['run', 't'])) is True
	assert _resolve_vision(parse(['run', 't', '--vision', 'off'])) is False
	assert _resolve_vision(parse(['run', 't', '--vision', 'auto'])) == 'auto'
	assert _resolve_vision(parse(['run', 't', '--no-vision'])) is False
	with pytest.raises(SystemExit, match='not both'):
		_resolve_vision(parse(['run', 't', '--vision', 'auto', '--no-vision']))


def test_ollama_needs_no_api_key(monkeypatch):
	for k in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']:
		monkeypatch.delenv(k, raising=False)
	assert resolve_llm('ollama:qwen2.5').model == 'qwen2.5'
	assert resolve_llm('ollama').model == 'llama3.1'


def test_vault_cli_never_prints_values(tmp_path, monkeypatch, capsys):
	monkeypatch.setenv('BU_VAULT_DIR', str(tmp_path))
	monkeypatch.setenv('MY_PW', 'p@ss-secret-123')
	assert main(['vault', 'set', 'example.com', 'password', '--from-env', 'MY_PW']) == 0
	assert main(['vault', 'list']) == 0
	out = capsys.readouterr().out
	assert 'example.com' in out and 'password' in out and 'p@ss-secret-123' not in out
	assert main(['vault', 'delete', 'example.com', 'password']) == 0
	with pytest.raises(SystemExit, match='not found'):
		main(['vault', 'delete', 'example.com', 'password'])

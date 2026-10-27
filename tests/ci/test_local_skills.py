"""Local skills: record a successful run once, find it by intent, replay with new parameters."""

import pytest

from browser_use.skills.local import LocalSkillStore


def test_save_load_list_roundtrip(tmp_path, make_history):
	store = LocalSkillStore(tmp_path)
	skill = store.save('login-flow', make_history(), task='Log in to the demo site', sensitive_data=None)
	assert skill.steps == 3
	assert 'email' in skill.parameters and skill.parameters['email'].example == 'me@example.com'

	assert [s.name for s in store.list()] == ['login-flow']
	assert store.load('login-flow').task == 'Log in to the demo site'
	md = (tmp_path / 'login-flow' / 'SKILL.md').read_text()
	assert md.startswith('---\nname: login-flow') and 'navigate(' in md
	assert (tmp_path / 'login-flow' / 'history.json').exists()


def test_secret_values_are_not_stored_as_examples(tmp_path, make_history):
	store = LocalSkillStore(tmp_path)
	skill = store.save('s', make_history(), task='t', sensitive_data={'x_email': 'me@example.com'})
	assert skill.parameters['email'].example is None
	assert 'me@example.com' not in (tmp_path / 's' / 'history.json').read_text()
	assert 'me@example.com' not in (tmp_path / 's' / 'SKILL.md').read_text()


def test_rejects_unsuccessful_runs_duplicates_and_bad_names(tmp_path, make_history):
	store = LocalSkillStore(tmp_path)
	with pytest.raises(ValueError, match='successfully'):
		store.save('bad', make_history(success=False), task='t')
	store.save('ok', make_history(), task='t')
	with pytest.raises(FileExistsError):
		store.save('ok', make_history(), task='t')
	store.save('ok', make_history(), task='t', overwrite=True)
	for name in ['../escape', 'A B', '', '/abs']:
		with pytest.raises(ValueError):
			store.save(name, make_history(), task='t')
	assert not (tmp_path.parent / 'escape').exists()


def test_find_ranks_by_intent(tmp_path, make_history):
	store = LocalSkillStore(tmp_path)
	store.save('login-flow', make_history(), task='Log in to the demo site', description='Sign in to the demo site account')
	store.save(
		'invoice-export', make_history(), task='Download last month invoices as PDF', description='Export invoices from billing'
	)
	ranked = store.find('please download my invoices')
	assert ranked and ranked[0][0].name == 'invoice-export'
	assert store.find('completely unrelated zebra') == []


async def test_run_rejects_unknown_parameters(tmp_path, make_history):
	from browser_use.llm.openai.chat import ChatOpenAI

	store = LocalSkillStore(tmp_path)
	store.save('login-flow', make_history(), task='Log in', sensitive_data=None)
	with pytest.raises(ValueError, match='Unknown parameters'):
		await store.run('login-flow', llm=ChatOpenAI(model='gpt-4o', api_key='unused'), variables={'nope': 'x'})
	store.delete('login-flow')
	assert store.list() == []

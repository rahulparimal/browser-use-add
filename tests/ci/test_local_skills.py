"""Local skills: record a successful run once, find it by intent, replay with new parameters."""

import pytest

from browser_use.agent.views import ActionResult, AgentHistory, AgentHistoryList, AgentOutput
from browser_use.browser.views import BrowserStateHistory
from browser_use.dom.views import DOMInteractedElement, NodeType
from browser_use.skills.local import LocalSkillStore
from browser_use.tools.service import Tools


def _history(success: bool = True) -> AgentHistoryList:
	tools = Tools()
	ActionModel = tools.registry.create_action_model()
	output_model = AgentOutput.type_with_custom_actions(ActionModel)
	email_input = DOMInteractedElement(
		node_id=1,
		backend_node_id=1,
		frame_id='f',
		node_type=NodeType.ELEMENT_NODE,
		node_value='',
		node_name='input',
		attributes={'type': 'email', 'id': 'email'},
		bounds=None,
		x_path='//*[@id="email"]',
		element_hash=1,
	)

	def item(action: dict, result: ActionResult, element=None) -> AgentHistory:
		return AgentHistory(
			model_output=output_model(action=[ActionModel(**action)]),
			result=[result],
			state=BrowserStateHistory(
				url='http://x.test', title='t', tabs=[], interacted_element=[element], screenshot_path=None
			),
		)

	return AgentHistoryList(
		history=[
			item({'navigate': {'url': 'http://x.test/login', 'new_tab': False}}, ActionResult()),
			item({'input': {'index': 1, 'text': 'me@example.com', 'clear': True}}, ActionResult(), email_input),
			item({'done': {'text': 'logged in', 'success': success}}, ActionResult(is_done=True, success=success)),
		]
	)


def test_save_load_list_roundtrip(tmp_path):
	store = LocalSkillStore(tmp_path)
	skill = store.save('login-flow', _history(), task='Log in to the demo site', sensitive_data=None)
	assert skill.steps == 3
	assert 'email' in skill.parameters and skill.parameters['email'].example == 'me@example.com'

	assert [s.name for s in store.list()] == ['login-flow']
	assert store.load('login-flow').task == 'Log in to the demo site'
	md = (tmp_path / 'login-flow' / 'SKILL.md').read_text()
	assert md.startswith('---\nname: login-flow') and 'navigate(' in md
	assert (tmp_path / 'login-flow' / 'history.json').exists()


def test_secret_values_are_not_stored_as_examples(tmp_path):
	store = LocalSkillStore(tmp_path)
	skill = store.save('s', _history(), task='t', sensitive_data={'x_email': 'me@example.com'})
	assert skill.parameters['email'].example is None
	assert 'me@example.com' not in (tmp_path / 's' / 'history.json').read_text()
	assert 'me@example.com' not in (tmp_path / 's' / 'SKILL.md').read_text()


def test_rejects_unsuccessful_runs_duplicates_and_bad_names(tmp_path):
	store = LocalSkillStore(tmp_path)
	with pytest.raises(ValueError, match='successfully'):
		store.save('bad', _history(success=False), task='t')
	store.save('ok', _history(), task='t')
	with pytest.raises(FileExistsError):
		store.save('ok', _history(), task='t')
	store.save('ok', _history(), task='t', overwrite=True)
	for name in ['../escape', 'A B', '', '/abs']:
		with pytest.raises(ValueError):
			store.save(name, _history(), task='t')
	assert not (tmp_path.parent / 'escape').exists()


def test_find_ranks_by_intent(tmp_path):
	store = LocalSkillStore(tmp_path)
	store.save('login-flow', _history(), task='Log in to the demo site', description='Sign in to the demo site account')
	store.save(
		'invoice-export', _history(), task='Download last month invoices as PDF', description='Export invoices from billing'
	)
	ranked = store.find('please download my invoices')
	assert ranked and ranked[0][0].name == 'invoice-export'
	assert store.find('completely unrelated zebra') == []


async def test_run_rejects_unknown_parameters(tmp_path):
	from browser_use.llm.openai.chat import ChatOpenAI

	store = LocalSkillStore(tmp_path)
	store.save('login-flow', _history(), task='Log in', sensitive_data=None)
	with pytest.raises(ValueError, match='Unknown parameters'):
		await store.run('login-flow', llm=ChatOpenAI(model='gpt-4o', api_key='unused'), variables={'nope': 'x'})
	store.delete('login-flow')
	assert store.list() == []

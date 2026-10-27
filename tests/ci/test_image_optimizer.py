"""Token-aware screenshot optimizer: sizing, encoding, dedupe, and wiring into the state message."""

import base64
import io

from PIL import Image, ImageDraw

from browser_use.agent.prompts import AgentMessagePrompt
from browser_use.browser.views import BrowserStateSummary, PageInfo, TabInfo
from browser_use.dom.views import SerializedDOMState
from browser_use.filesystem.file_system import FileSystem
from browser_use.screenshots.optimizer import (
	ImageOptimizer,
	ImageOptimizerSettings,
	detect_provider,
	estimate_image_tokens,
)


def _png_b64(width: int, height: int, text: str = 'hello') -> str:
	img = Image.new('RGB', (width, height), (240, 240, 240))
	draw = ImageDraw.Draw(img)
	for i in range(0, height, 40):
		draw.rectangle([20, i + 5, width // 2, i + 25], fill=(30, 60, 200))
		draw.text((30, i + 8), f'{text} {i}', fill=(255, 255, 255))
	buf = io.BytesIO()
	img.save(buf, format='PNG')
	return base64.b64encode(buf.getvalue()).decode()


def test_downscales_into_box_and_never_upscales():
	opt = ImageOptimizer(ImageOptimizerSettings(max_width=1280, max_height=800))
	big = opt.optimize(_png_b64(2560, 1600))
	assert (big.width, big.height) == (1280, 800)
	small = opt.optimize(_png_b64(640, 400))
	assert (small.width, small.height) == (640, 400)


def _photo_png_b64(width: int, height: int) -> str:
	img = Image.effect_noise((width, height), 30).convert('RGB')
	buf = io.BytesIO()
	img.save(buf, format='PNG')
	return base64.b64encode(buf.getvalue()).decode()


def test_lossy_encoding_shrinks_payload_and_sets_media_type():
	opt = ImageOptimizer(ImageOptimizerSettings(format='jpeg', quality=60))
	result = opt.optimize(_photo_png_b64(1920, 1080))
	assert result.media_type == 'image/jpeg'
	assert result.data_url.startswith('data:image/jpeg;base64,')
	assert result.optimized_bytes < result.original_bytes
	assert Image.open(io.BytesIO(base64.b64decode(result.data))).format == 'JPEG'


def test_skip_unchanged_frames_only_when_enabled():
	frame = _png_b64(1280, 800)
	off = ImageOptimizer(ImageOptimizerSettings())
	assert not off.optimize(frame).unchanged
	assert not off.optimize(frame).unchanged

	on = ImageOptimizer(ImageOptimizerSettings(skip_unchanged=True))
	assert not on.optimize(frame).unchanged
	assert on.optimize(frame).unchanged
	assert on.frames_skipped == 1
	assert not on.optimize(_photo_png_b64(1280, 800)).unchanged
	on.reset()
	assert not on.optimize(frame).unchanged


def test_never_inflates_flat_ui_renders():
	# flat colour UI compresses better as PNG than JPEG; for an untouched-size frame the optimizer must
	# pick the smallest encoding (resizing is judged on tokens, not bytes: antialiasing can add bytes)
	result = ImageOptimizer(ImageOptimizerSettings(format='jpeg')).optimize(_png_b64(1280, 720))
	assert result.optimized_bytes <= result.original_bytes


def test_memoizes_by_content():
	opt = ImageOptimizer(ImageOptimizerSettings(cache_size=2))
	frame = _png_b64(800, 600)
	assert opt.optimize(frame) is opt.optimize(frame)


def test_token_estimates_follow_provider_formulas():
	assert estimate_image_tokens(1000, 750, 'anthropic') == 1000
	# anthropic downscales above 1568px long edge, so huge images are capped
	assert estimate_image_tokens(4000, 4000, 'anthropic') <= 3300
	assert estimate_image_tokens(1024, 768, 'openai', 'low') == 85
	assert estimate_image_tokens(1024, 768, 'openai') == 85 + 170 * 4
	assert estimate_image_tokens(300, 300, 'google') == 258
	assert estimate_image_tokens(1536, 768, 'google') == 516


def test_optimizing_reduces_estimated_tokens_per_provider():
	for model, provider in [('claude-sonnet-4-6', 'anthropic'), ('gpt-4o', 'openai'), ('gemini-2.5-flash', 'google')]:
		assert detect_provider(model) == provider
		opt = ImageOptimizer(ImageOptimizerSettings.for_model(model))
		r = opt.optimize(_png_b64(2560, 1440))
		assert estimate_image_tokens(r.width, r.height, provider) < estimate_image_tokens(2560, 1440, provider)


def _prompt(tmp_path, screenshot: str | None, optimizer: ImageOptimizer | None, history_parts: list[str] | None = None):
	bs = BrowserStateSummary(
		url='https://example.test/foo',
		title='Test',
		tabs=[TabInfo(target_id='abcd1234', url='https://example.test/foo', title='Test')],
		page_info=PageInfo(
			viewport_width=1280,
			viewport_height=720,
			page_width=1280,
			page_height=1440,
			scroll_x=0,
			scroll_y=0,
			pixels_above=0,
			pixels_below=720,
			pixels_left=0,
			pixels_right=0,
		),
		dom_state=SerializedDOMState(_root=None, selector_map={}),
		is_pdf_viewer=False,
		recent_events=None,
		closed_popup_messages=[],
		screenshot=screenshot,
	)
	return AgentMessagePrompt(
		browser_state_summary=bs,
		file_system=FileSystem(base_dir=str(tmp_path), create_default_files=False),
		agent_history_description='\n'.join(history_parts or []),
		agent_history_parts=history_parts,
		task='Do the test thing',
		screenshots=[screenshot] if screenshot else [],
		image_optimizer=optimizer,
	)


def test_state_message_uses_optimized_image(tmp_path):
	opt = ImageOptimizer(ImageOptimizerSettings(skip_unchanged=True))
	frame = _photo_png_b64(1920, 1080)

	first = _prompt(tmp_path, frame, opt).get_user_message(use_vision=True)
	images = [p for p in first.content if p.type == 'image_url']
	assert len(images) == 1 and images[0].image_url.media_type == 'image/jpeg'

	second = _prompt(tmp_path, frame, opt).get_user_message(use_vision=True)
	assert not [p for p in second.content if p.type == 'image_url']
	assert 'unchanged' in second.text


def test_history_chunks_carry_the_only_cache_breakpoint(tmp_path):
	parts = ['<step>one</step>', '<step>two</step>', '<step>three</step>']
	msg = _prompt(tmp_path, None, None, history_parts=parts).get_user_message(use_vision=False)
	assert msg.cache is False  # volatile tail must not be cached (write cost, never read)
	flagged = [i for i, p in enumerate(msg.content) if p.cache]
	assert flagged == [2]  # last history chunk
	text = ''.join(p.text for p in msg.content)
	assert text.index('<step>three</step>') < text.index('</agent_history>') < text.index('<browser_state>')


def test_anthropic_serializer_emits_per_part_cache_control(tmp_path):
	from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer

	parts = ['<step>one</step>', '<step>two</step>']
	msg = _prompt(tmp_path, None, None, history_parts=parts).get_user_message(use_vision=False)
	serialized, _ = AnthropicMessageSerializer.serialize_messages([msg])
	blocks = serialized[0]['content']
	cached = [i for i, b in enumerate(blocks) if b.get('cache_control')]
	assert cached == [1]
	assert len(blocks) == 3

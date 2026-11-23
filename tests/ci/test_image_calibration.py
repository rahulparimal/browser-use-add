"""Measured image-token calibration replaces stale published formulas."""

from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from browser_use.screenshots.calibration import (
	ImageTokenCalibration,
	calibrate_image_tokens,
	fit,
	load_calibration,
	save_calibration,
)
from browser_use.screenshots.optimizer import ImageOptimizerSettings, estimate_image_tokens


def test_fit_and_estimate():
	cal = fit('m', [(512 * 384, 2000), (1024 * 768, 8000)])
	assert cal.estimate(1024, 768) == 8000
	assert cal.estimate(512, 384) == 2000
	assert cal.estimate(768, 576) < cal.estimate(1024, 768)


def test_save_load_roundtrip_per_model(tmp_path):
	path = tmp_path / 'cal.json'
	save_calibration(ImageTokenCalibration(model='a', base_tokens=1, tokens_per_pixel=0.01), path)
	save_calibration(ImageTokenCalibration(model='b', base_tokens=2, tokens_per_pixel=0.02), path)
	assert load_calibration('a', path).tokens_per_pixel == 0.01
	assert load_calibration('b', path).base_tokens == 2
	assert load_calibration('missing', path) is None
	assert load_calibration('a', tmp_path / 'nope.json') is None


def test_estimate_prefers_calibration_over_formula():
	cal = ImageTokenCalibration(model='gpt-6-luna', base_tokens=0, tokens_per_pixel=8000 / (1024 * 768))
	assert estimate_image_tokens(1024, 768, 'openai') == 765  # old published formula
	assert estimate_image_tokens(1024, 768, 'openai', calibration=cal) == 8000


def test_openai_default_box_is_smaller():
	s = ImageOptimizerSettings.for_model('gpt-4o')
	assert (s.max_width, s.max_height) == (768, 576)


class _PixelBilledLLM:
	"""Real object implementing the chat protocol; bills 0.01 tokens/pixel for images (the allowed LLM stand-in)."""

	model = 'fake'
	_verified_api_keys = True
	provider = name = 'fake'

	async def ainvoke(self, messages, output_format=None, **kwargs):
		tokens = 10
		for part in messages[0].content:
			if part.type == 'image_url':
				import base64
				import io

				from PIL import Image

				img = Image.open(io.BytesIO(base64.b64decode(part.image_url.url.split(',')[1])))
				tokens += round(img.width * img.height * 0.01)
		return ChatInvokeCompletion(
			completion='ok',
			usage=ChatInvokeUsage(
				prompt_tokens=tokens,
				prompt_cached_tokens=None,
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=None,
				completion_tokens=1,
				total_tokens=tokens + 1,
			),
		)


async def test_calibrate_recovers_the_billing_rate():
	cal = await calibrate_image_tokens(_PixelBilledLLM())
	assert abs(cal.tokens_per_pixel - 0.01) < 1e-4
	assert abs(cal.estimate(1024, 768) - 7864) <= 2

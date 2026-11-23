"""Measure what a model really charges per image, instead of trusting a published formula.

Providers change image pricing per model generation, so `estimate_image_tokens` formulas go stale. Calibration sends
a text-only prompt and the same prompt with synthetic images of two sizes, takes the `prompt_tokens` difference and
fits `tokens = base + per_pixel * pixels`. Results are stored per model and used by `estimate_image_tokens`.
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel, ConfigDict

from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, UserMessage

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel

_SIZES = ((512, 384), (1024, 768))
_PROMPT = 'Reply with the single word: ok'


class ImageTokenCalibration(BaseModel):
	model_config = ConfigDict(frozen=True)

	model: str
	base_tokens: float
	tokens_per_pixel: float

	def estimate(self, width: int, height: int) -> int:
		return max(0, round(self.base_tokens + self.tokens_per_pixel * width * height))


def fit(model: str, points: list[tuple[int, int]]) -> ImageTokenCalibration:
	"""Least-squares through two (pixels, image_tokens) points."""
	assert len(points) == 2 and points[0][0] != points[1][0], 'need two distinct sizes'
	(p1, t1), (p2, t2) = points
	slope = (t2 - t1) / (p2 - p1)
	return ImageTokenCalibration(model=model, base_tokens=t1 - slope * p1, tokens_per_pixel=slope)


def calibration_path() -> Path:
	from browser_use.config import CONFIG

	return Path(os.environ.get('BU_IMAGE_CALIBRATION_FILE') or CONFIG.BROWSER_USE_CONFIG_DIR / 'image_token_calibration.json')


def load_calibration(model: str, path: Path | None = None) -> ImageTokenCalibration | None:
	path = path or calibration_path()
	try:
		raw = json.loads(path.read_text(encoding='utf-8'))
		return ImageTokenCalibration.model_validate(raw[model])
	except (OSError, KeyError, ValueError):
		return None


def save_calibration(cal: ImageTokenCalibration, path: Path | None = None) -> Path:
	path = path or calibration_path()
	try:
		existing = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, ValueError):
		existing = {}
	existing[cal.model] = cal.model_dump()
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
	return path


def _noise_png_b64(width: int, height: int) -> str:
	# noise defeats any server-side downscaling/cropping heuristics tied to content
	buf = BytesIO()
	Image.effect_noise((width, height), 64).convert('RGB').save(buf, format='PNG')
	return base64.b64encode(buf.getvalue()).decode()


async def _prompt_tokens(llm: BaseChatModel, image: tuple[int, int] | None) -> int:
	parts: list[ContentPartTextParam | ContentPartImageParam] = [ContentPartTextParam(text=_PROMPT)]
	if image:
		parts.append(
			ContentPartImageParam(
				image_url=ImageURL(url=f'data:image/png;base64,{_noise_png_b64(*image)}', media_type='image/png', detail='auto')
			)
		)
	result = await llm.ainvoke([UserMessage(content=parts)])
	assert result.usage is not None, 'model returned no usage; cannot calibrate'
	return result.usage.prompt_tokens


async def calibrate_image_tokens(llm: BaseChatModel) -> ImageTokenCalibration:
	"""3 tiny API calls (one text-only, two with images). Costs a fraction of a cent on most models."""
	baseline = await _prompt_tokens(llm, None)
	points = [(w * h, await _prompt_tokens(llm, (w, h)) - baseline) for w, h in _SIZES]
	return fit(llm.model, points)

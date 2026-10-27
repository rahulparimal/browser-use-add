"""Token-aware screenshot optimizer.

Vision tokens are billed by pixel geometry, not by byte size, so the levers that matter are:

1. Resolution: fit the image into the provider's sweet-spot box (never upscale).
2. Redundancy: skip frames that are perceptually identical to the one the model already saw.
3. Transport: lossy encoding (JPEG/WebP) cuts request bytes/latency 3-10x for UI screenshots.

`estimate_image_tokens` follows the published formulas of each provider so the optimizer (and callers)
can reason about cost without a round-trip.
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections import OrderedDict
from io import BytesIO
from typing import Annotated, Literal

from PIL import Image
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from browser_use.llm.messages import SupportedImageMediaType

Provider = Literal['anthropic', 'openai', 'google', 'other']

_MEDIA_TYPES: dict[str, SupportedImageMediaType] = {'jpeg': 'image/jpeg', 'webp': 'image/webp', 'png': 'image/png'}


def _positive(v: int) -> int:
	assert v > 0, 'must be > 0'
	return v


def detect_provider(model_name: str) -> Provider:
	"""Best-effort provider detection from a model id (handles gateway ids like 'anthropic/claude-...')."""
	name = model_name.lower().rsplit('/', 1)[-1]
	if name.startswith('claude'):
		return 'anthropic'
	if name.startswith(('gpt', 'o1', 'o3', 'o4', 'chatgpt')):
		return 'openai'
	if name.startswith('gemini'):
		return 'google'
	return 'other'


def estimate_image_tokens(width: int, height: int, provider: Provider = 'other', detail: str = 'auto') -> int:
	"""Estimate input tokens for an image of the given size.

	- anthropic: ~ width*height/750 (long edge is downscaled by the API above ~1568px)
	- openai: 'low' = 85; otherwise fit in 2048 box, shortest side -> 768, then 170/512px-tile + 85
	- google: 258 if both sides <= 384, otherwise 258 per 768px tile
	"""
	assert width > 0 and height > 0
	if provider == 'anthropic':
		scale = min(1.0, 1568 / max(width, height))
		return math.ceil(width * scale * height * scale / 750)
	if provider == 'openai':
		if detail == 'low':
			return 85
		scale = min(1.0, 2048 / max(width, height))
		w, h = width * scale, height * scale
		scale = min(1.0, 768 / min(w, h))
		w, h = w * scale, h * scale
		return 85 + 170 * math.ceil(w / 512) * math.ceil(h / 512)
	if provider == 'google':
		if width <= 384 and height <= 384:
			return 258
		return 258 * math.ceil(width / 768) * math.ceil(height / 768)
	return math.ceil(width * height / 750)


# (max_width, max_height) boxes where each provider's price curve is flat-ish for UI screenshots.
_PROVIDER_BOX: dict[Provider, tuple[int, int]] = {
	'anthropic': (1280, 800),  # ~1.4k tokens vs ~2k at 1568 long edge
	'openai': (1024, 768),  # 2x2 tiles = 765 tokens
	'google': (1024, 768),  # 2x1 tiles = 516 tokens
	'other': (1280, 800),
}


class ImageOptimizerSettings(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	max_width: Annotated[int, AfterValidator(_positive)] = 1280
	max_height: Annotated[int, AfterValidator(_positive)] = 800
	format: Literal['jpeg', 'webp', 'png'] = 'jpeg'
	quality: Annotated[int, Field(ge=10, le=100)] = 70
	skip_unchanged: bool = False
	"""Drop a screenshot that is perceptually identical to the previous one sent (DOM text is still sent)."""
	unchanged_hamming_distance: Annotated[int, Field(ge=0, le=64)] = 2
	"""Max dHash bit distance (of 64) still treated as 'unchanged'."""
	cache_size: Annotated[int, Field(ge=1)] = 8

	@classmethod
	def for_model(cls, model_name: str, **overrides) -> ImageOptimizerSettings:
		w, h = _PROVIDER_BOX[detect_provider(model_name)]
		return cls(max_width=w, max_height=h, **overrides)


class OptimizedImage(BaseModel):
	model_config = ConfigDict(frozen=True)

	data: str  # base64, no data: prefix
	media_type: SupportedImageMediaType
	width: int
	height: int
	original_bytes: int
	optimized_bytes: int
	fingerprint: int  # 64-bit dHash of the *input* frame
	unchanged: bool = False  # True when skip_unchanged is on and the frame matches the previous one

	@property
	def data_url(self) -> str:
		return f'data:{self.media_type};base64,{self.data}'


def dhash(img: Image.Image) -> int:
	"""64-bit difference hash: robust to scaling/compression, cheap (9x8 grayscale)."""
	small = img.convert('L').resize((9, 8), Image.Resampling.BILINEAR)
	px = list(small.tobytes())  # 'L' mode: one byte per pixel
	bits = 0
	for row in range(8):
		for col in range(8):
			bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
	return bits


class ImageOptimizer:
	"""Stateful optimizer: memoizes results by content hash and remembers the last frame fingerprint."""

	def __init__(self, settings: ImageOptimizerSettings | None = None):
		self.settings = settings or ImageOptimizerSettings()
		self._cache: OrderedDict[str, OptimizedImage] = OrderedDict()
		self._last_fingerprint: int | None = None
		self.total_original_bytes = 0
		self.total_optimized_bytes = 0
		self.frames_skipped = 0

	def optimize(self, screenshot_b64: str) -> OptimizedImage:
		key = hashlib.blake2b(screenshot_b64.encode(), digest_size=16).hexdigest()
		cached = self._cache.get(key)
		if cached is None:
			cached = self._encode(screenshot_b64)
			self._cache[key] = cached
			while len(self._cache) > self.settings.cache_size:
				self._cache.popitem(last=False)
		else:
			self._cache.move_to_end(key)

		unchanged = (
			self.settings.skip_unchanged
			and self._last_fingerprint is not None
			and (self._last_fingerprint ^ cached.fingerprint).bit_count() <= self.settings.unchanged_hamming_distance
		)
		self._last_fingerprint = cached.fingerprint
		if unchanged:
			self.frames_skipped += 1
			return cached.model_copy(update={'unchanged': True})
		self.total_original_bytes += cached.original_bytes
		self.total_optimized_bytes += cached.optimized_bytes
		return cached

	def reset(self) -> None:
		"""Forget the previous frame (e.g. after compaction, when the model no longer has it in context)."""
		self._last_fingerprint = None

	def _encode(self, screenshot_b64: str) -> OptimizedImage:
		raw = base64.b64decode(screenshot_b64)
		s = self.settings
		img = Image.open(BytesIO(raw))
		img.load()
		fingerprint = dhash(img)

		scale = min(1.0, s.max_width / img.width, s.max_height / img.height)
		if scale < 1.0:
			img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)

		def png_bytes() -> bytes:
			b = BytesIO()
			img.save(b, format='PNG', optimize=True)
			return b.getvalue()

		if s.format == 'png':
			out, media_type = png_bytes(), 'image/png'
		else:
			lossy_img = img
			if img.mode not in ('RGB', 'L'):
				# flatten alpha onto white; lossy screenshots have no meaningful transparency
				rgba = img.convert('RGBA')
				lossy_img = Image.new('RGB', img.size, (255, 255, 255))
				lossy_img.paste(rgba, mask=rgba.split()[-1])
			b = BytesIO()
			lossy_img.save(b, format=s.format.upper(), quality=s.quality, optimize=True)
			out, media_type = b.getvalue(), _MEDIA_TYPES[s.format]
			# Flat UI renders (few colors, sharp edges) can compress better as PNG than as JPEG/WebP;
			# only pay for the PNG pass when the lossy result is not already clearly small.
			if len(out) > len(raw) // 4:
				png = png_bytes()
				if len(png) < len(out):
					out, media_type = png, 'image/png'
		if scale == 1.0 and len(out) >= len(raw):  # never make an untouched frame bigger
			out, media_type = raw, 'image/png'
		return OptimizedImage(
			data=base64.b64encode(out).decode(),
			media_type=media_type,
			width=img.width,
			height=img.height,
			original_bytes=len(raw),
			optimized_bytes=len(out),
			fingerprint=fingerprint,
		)

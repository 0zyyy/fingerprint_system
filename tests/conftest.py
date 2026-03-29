"""
Shared fixtures and mock setup for the test suite.

Hardware mocking strategy:
  - RPi.GPIO     → fake-rpi (already a dep) + module-level patch
  - adafruit_fingerprint → fully mocked (no hardware available)
  - onnxruntime  → InferenceSession patched to return controlled outputs
  - serial.Serial → patched to avoid real UART
"""
import sys
import types
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub out hardware-only modules before any project code is imported
# ---------------------------------------------------------------------------

def _make_adafruit_stub():
    mod = types.ModuleType("adafruit_fingerprint")
    mod.OK = 0
    mod.NOFINGER = 2
    mod.Adafruit_Fingerprint = MagicMock()
    return mod


def _make_rpi_stub():
    rpi = types.ModuleType("RPi")
    gpio = types.ModuleType("RPi.GPIO")
    gpio.BOARD = 11
    gpio.OUT = 0
    gpio.HIGH = 1
    gpio.LOW = 0
    gpio.setmode = MagicMock()
    gpio.setwarnings = MagicMock()
    gpio.setup = MagicMock()
    gpio.output = MagicMock()
    gpio.cleanup = MagicMock()
    gpio.PWM = MagicMock(return_value=MagicMock())
    rpi.GPIO = gpio
    return rpi, gpio


_rpi, _gpio = _make_rpi_stub()
sys.modules.setdefault("RPi", _rpi)
sys.modules.setdefault("RPi.GPIO", _gpio)
sys.modules.setdefault("adafruit_fingerprint", _make_adafruit_stub())


# ---------------------------------------------------------------------------
# Synthetic fingerprint image helpers
# ---------------------------------------------------------------------------

def make_fingerprint_image(seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """
    Generate a 288×256 grayscale image that mimics fingerprint ridge patterns.
    Deterministic per seed; optional noise for robustness tests.
    """
    rng = np.random.default_rng(seed)
    h, w = 288, 256
    y, x = np.mgrid[0:h, 0:w]

    # Concentric ellipses → ridge-like pattern
    cx, cy = w // 2, h // 2
    dist = np.sqrt(((x - cx) / (w * 0.4)) ** 2 + ((y - cy) / (h * 0.4)) ** 2)
    ridges = (np.sin(dist * 20) * 127 + 128).astype(np.uint8)

    if noise > 0:
        ridges = np.clip(
            ridges.astype(np.float32) + rng.normal(0, noise * 255, ridges.shape),
            0, 255,
        ).astype(np.uint8)

    return ridges


def make_hog_vector(seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """Return a 1764-dim HOG vector extracted from a synthetic image."""
    from core.preprocessing import ImagePreprocessor
    from core.feature_extraction import HOGExtractor

    img = make_fingerprint_image(seed=seed, noise=noise)
    processed = ImagePreprocessor().process(img)
    return HOGExtractor().extract(processed)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """DatabaseManager backed by a temp SQLite file."""
    monkeypatch.setitem(
        __import__("config").SYSTEM_CONFIG,
        "DATABASE_PATH",
        str(tmp_path / "test.db"),
    )
    from database.db_manager import DatabaseManager
    return DatabaseManager()


@pytest.fixture
def sample_image():
    return make_fingerprint_image(seed=42)


@pytest.fixture
def sample_vector():
    return make_hog_vector(seed=42)

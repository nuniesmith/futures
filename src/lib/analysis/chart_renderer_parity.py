"""
Chart Renderer Parity — C#-Matching CNN Snapshot Renderer
==========================================================
Renders 224×224 chart images using pure Pillow (no mplfinance) that
pixel-match the C# ``OrbChartRenderer`` in ``BreakoutStrategy.cs``.

**Why this exists:**
The biggest accuracy blocker in the ORB CNN pipeline is the distribution
shift between training images (rendered by ``chart_renderer.py`` via
mplfinance with EMA9 lines, quality badges, legends, and matplotlib
anti-aliasing) and inference images (rendered by the C# ``OrbChartRenderer``
using System.Drawing with pixel-precise integer coordinates and no AA).

This module replicates the *exact* C# rendering logic — same canvas size,
same padding, same color values, same coordinate math, same volume panel
height — so that training images are visually identical to what the model
sees at inference time in NinjaTrader.

**Rendering contract (must match BreakoutStrategy.cs OrbChartRenderer):**
  - Canvas:    224 × 224 pixels
  - VolPanel:  40 pixels at the bottom
  - PriceH:    224 - 40 - 4 = 180 pixels (price chart area)
  - PriceTop:  4 pixels
  - LeftPad:   4 pixels
  - RightPad:  4 pixels
  - Background:    RGB(13, 13, 13)     = #0D0D0D
  - Bull candle:   RGB(38, 166, 154)   = #26A69A
  - Bear candle:   RGB(239, 83, 80)    = #EF5350
  - ORB fill:      RGBA(255, 215, 0, 40)
  - ORB border:    RGBA(255, 215, 0, 100)
  - VWAP line:     RGB(0, 229, 255)    = #00E5FF
  - Vol bull:      RGBA(38, 166, 154, 100)
  - Vol bear:      RGBA(239, 83, 80, 100)

**C# reference (OrbChartRenderer.Render method logic):**
  1. Fill background
  2. Compute price min/max from bar data + ORB levels
  3. Compute volume max
  4. Draw ORB fill rectangle + border lines
  5. Draw VWAP line across all bars
  6. Draw volume bars (bottom panel)
  7. Draw candlestick bodies + wicks

The coordinate math is replicated line-for-line from the C# source.

Public API:
    from chart_renderer_parity import (
        render_parity_snapshot,
        render_parity_to_temp,
        ParityBar,
    )

Dependencies:
    - Pillow >= 10.0.0 (already in project)
    - numpy (already in project)
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("analysis.chart_renderer_parity")

try:
    from PIL import Image, ImageDraw

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning("Pillow not installed — parity chart rendering disabled")


# ---------------------------------------------------------------------------
# Constants — MUST match OrbChartRenderer in BreakoutStrategy.cs exactly
# ---------------------------------------------------------------------------

W = 224
H = 224
VOL_PANEL_H = 40
PRICE_H = H - VOL_PANEL_H - 4  # 180
PRICE_TOP = 4
LEFT_PAD = 4
RIGHT_PAD = 4

# Colors — exact RGB/RGBA from C# source
BG_COLOR = (13, 13, 13)  # Color.FromArgb(0x0D, 0x0D, 0x0D)
BULL_CANDLE = (38, 166, 154)  # Color.FromArgb(0x26, 0xA6, 0x9A)
BEAR_CANDLE = (239, 83, 80)  # Color.FromArgb(0xEF, 0x53, 0x50)
VWAP_LINE = (0, 229, 255)  # Color.FromArgb(0x00, 0xE5, 0xFF)
VOL_BULL = (38, 166, 154, 100)  # Color.FromArgb(100, 0x26, 0xA6, 0x9A)
VOL_BEAR = (239, 83, 80, 100)  # Color.FromArgb(100, 0xEF, 0x53, 0x50)

# ---------------------------------------------------------------------------
# Per-BreakoutType box colors — must match C# OrbChartRenderer style tokens
# ---------------------------------------------------------------------------
# Each entry is (fill_rgba, border_rgba, dashed).
# "dashed" True → drawn with on/off segments matching C# DashStyle.Dash.
# "dashed" False → solid line.
#
# C# reference (BreakoutStrategy.cs OrbChartRenderer):
#   ORB          — Gold    dashed  — Color.FromArgb(30/100, 0xFF, 0xD7, 0x00)
#   PrevDay      — Silver  solid   — Color.FromArgb(20/120, 0xC0, 0xC0, 0xC0)
#   InitialBal.  — Cyan    dashed  — Color.FromArgb(18/110, 0x00, 0xE5, 0xFF)
#   Consolidation— Purple  solid   — Color.FromArgb(22/130, 0x93, 0x00, 0xD3)

# ORB — gold dashed (legacy constant names kept for backward compat)
ORB_FILL = (255, 215, 0, 30)  # Color.FromArgb(30,  0xFF, 0xD7, 0x00)
ORB_BORDER = (255, 215, 0, 100)  # Color.FromArgb(100, 0xFF, 0xD7, 0x00)

# PrevDay — silver solid
PREV_DAY_FILL = (192, 192, 192, 20)  # Color.FromArgb(20,  0xC0, 0xC0, 0xC0)
PREV_DAY_BORDER = (192, 192, 192, 120)  # Color.FromArgb(120, 0xC0, 0xC0, 0xC0)

# InitialBalance — cyan dashed
IB_FILL = (0, 229, 255, 18)  # Color.FromArgb(18,  0x00, 0xE5, 0xFF)
IB_BORDER = (0, 229, 255, 110)  # Color.FromArgb(110, 0x00, 0xE5, 0xFF)

# Consolidation — purple solid
CONSOL_FILL = (147, 0, 211, 22)  # Color.FromArgb(22,  0x93, 0x00, 0xD3)
CONSOL_BORDER = (147, 0, 211, 130)  # Color.FromArgb(130, 0x93, 0x00, 0xD3)

# box_style token → (fill_rgba, border_rgba, dashed)
_BOX_STYLES: dict[str, tuple[tuple, tuple, bool]] = {
    "gold_dashed": (ORB_FILL, ORB_BORDER, True),
    "silver_solid": (PREV_DAY_FILL, PREV_DAY_BORDER, False),
    "blue_dashed": (IB_FILL, IB_BORDER, True),
    "purple_solid": (CONSOL_FILL, CONSOL_BORDER, False),
}
_DEFAULT_BOX_STYLE = "gold_dashed"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParityBar:
    """Single OHLCV bar — mirrors OrbChartRenderer.Bar in C#."""

    open: float
    high: float
    low: float
    close: float
    volume: float

    def __init__(
        self,
        open: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
        close: float = 0.0,
        volume: float = 0.0,
    ):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


# ---------------------------------------------------------------------------
# Coordinate helpers — replicate C# integer math exactly
# ---------------------------------------------------------------------------


def _resolve_box_style(breakout_type: str | None) -> tuple[tuple, tuple, bool]:
    """Return ``(fill_rgba, border_rgba, dashed)`` for *breakout_type*.

    Accepts either a ``box_style`` token (e.g. ``"gold_dashed"``) or a
    ``BreakoutType`` name (e.g. ``"ORB"``, ``"PrevDay"``).  Falls back to
    ORB gold-dashed for unknown values.
    """
    if breakout_type is None:
        return _BOX_STYLES[_DEFAULT_BOX_STYLE]

    _bt = breakout_type.strip().lower()

    # Direct box_style token lookup
    for key, val in _BOX_STYLES.items():
        if _bt == key.lower():
            return val

    # BreakoutType name → box_style mapping
    _name_map = {
        "orb": "gold_dashed",
        "prevday": "silver_solid",
        "prev_day": "silver_solid",
        "initialbalance": "blue_dashed",
        "initial_balance": "blue_dashed",
        "consolidation": "purple_solid",
    }
    style_key = _name_map.get(_bt, _DEFAULT_BOX_STYLE)
    return _BOX_STYLES[style_key]


def _draw_dashed_hline(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    color: tuple,
    dash_on: int = 6,
    dash_off: int = 4,
) -> None:
    """Draw a horizontal dashed line on *draw* from x0 to x1 at row y.

    Replicates C# ``DashStyle.Dash`` segments.

    Args:
        draw:     Pillow ``ImageDraw`` instance.
        x0, x1:  Start and end x coordinates.
        y:        Row y coordinate.
        color:    RGBA or RGB color tuple.
        dash_on:  Number of pixels in the "on" (drawn) segment.
        dash_off: Number of pixels in the "off" (gap) segment.
    """
    x = x0
    cycle = dash_on + dash_off
    while x < x1:
        seg_end = min(x + dash_on, x1)
        draw.line([(x, y), (seg_end, y)], fill=color, width=1)
        x += cycle


def _price_to_y(price: float, price_min: float, price_range: float) -> int:
    """Map a price value to a Y pixel coordinate in the price panel.

    Mirrors the C# logic:
        int y = PriceTop + (int)((priceMax - price) / priceRange * PriceH);

    Note: C# (int) cast truncates toward zero, same as Python int() for
    positive values. Since (priceMax - price) >= 0 and priceRange > 0,
    the result is always non-negative, so int() matches (int) exactly.
    """
    if price_range <= 0:
        return PRICE_TOP + PRICE_H // 2
    # priceMax = price_min + price_range
    price_max = price_min + price_range
    y = PRICE_TOP + int((price_max - price) / price_range * PRICE_H)
    return y


def _vol_to_h(vol: float, vol_max: float) -> int:
    """Map a volume value to a height in the volume panel.

    Mirrors the C# logic:
        int vh = (int)(bar.Volume / volMax * VolPanelH);
    """
    if vol_max <= 0:
        return 0
    return int(vol / vol_max * VOL_PANEL_H)


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------


def render_parity_snapshot(
    bars: Sequence[ParityBar],
    orb_high: float,
    orb_low: float,
    vwap_values: Sequence[float] | None = None,
    direction: str | None = None,
    breakout_type: str | None = None,
) -> Image.Image | None:
    """Render a 224×224 chart image matching C# OrbChartRenderer.Render().

    This replicates the exact rendering pipeline from the C# method
    ``OrbChartRenderer.Render(Bar[] bars, double orbHigh, double orbLow,
    double[] vwap, string direction, BreakoutType btype)`` line-for-line.

    Args:
        bars: Sequence of ParityBar objects (OHLCV).
        orb_high: Opening range high level.
        orb_low: Opening range low level.
        vwap_values: Per-bar VWAP values (same length as bars).
                     If None, VWAP line is not drawn.
        direction: "long" or "short" (currently unused in C# renderer
                   but included for API parity).
        breakout_type: Box style selector.  Accepts a ``BreakoutType``
                       name (``"ORB"``, ``"PrevDay"``, ``"InitialBalance"``,
                       ``"Consolidation"``) or a ``box_style`` token
                       (``"gold_dashed"``, ``"silver_solid"``, etc.).
                       Defaults to ``"gold_dashed"`` (ORB) when None.

    Returns:
        PIL Image (224×224 RGB), or None if rendering failed.
    """
    if not _PIL_AVAILABLE:
        logger.warning("Pillow not available — cannot render parity chart")
        return None

    if not bars or len(bars) == 0:
        logger.warning("No bars provided — cannot render")
        return None

    n = len(bars)

    # ── Step 1: Create canvas and fill background ──────────────────────
    # In C#: using (var bmp = new Bitmap(W, H))
    #        using (var g = Graphics.FromImage(bmp))
    #        g.Clear(BgColor);
    img = Image.new("RGB", (W, H), BG_COLOR)

    # We need an RGBA overlay for semi-transparent elements (ORB fill,
    # volume bars). We'll composite at the end.
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_main = ImageDraw.Draw(img)
    draw_overlay = ImageDraw.Draw(overlay)

    # ── Step 2: Compute price range from bars + ORB levels ─────────────
    # C#:
    #   double priceMin = bars.Min(b => b.Low);
    #   double priceMax = bars.Max(b => b.High);
    #   priceMin = Math.Min(priceMin, orbLow);
    #   priceMax = Math.Max(priceMax, orbHigh);
    #   double priceRange = priceMax - priceMin;
    #   if (priceRange <= 0) priceRange = 1;
    price_min = min(b.low for b in bars)
    price_max = max(b.high for b in bars)
    price_min = min(price_min, orb_low)
    price_max = max(price_max, orb_high)
    price_range = price_max - price_min
    if price_range <= 0:
        price_range = 1.0

    # ── Step 3: Compute volume max ─────────────────────────────────────
    # C#:
    #   double volMax = bars.Max(b => b.Volume);
    #   if (volMax <= 0) volMax = 1;
    vol_max = max(b.volume for b in bars)
    if vol_max <= 0:
        vol_max = 1.0

    # ── Step 4: Compute bar geometry ───────────────────────────────────
    # C#:
    #   int usableW = W - LeftPad - RightPad;
    #   int barW = Math.Max(1, usableW / bars.Length);
    #   int bodyW = Math.Max(1, barW - 2);
    usable_w = W - LEFT_PAD - RIGHT_PAD
    bar_w = max(1, usable_w // n)
    body_w = max(1, bar_w - 2)

    # ── Step 5: Resolve box colors for this BreakoutType ───────────────
    box_fill, box_border, box_dashed = _resolve_box_style(breakout_type)

    # ── Step 6: Draw range fill rectangle ──────────────────────────────
    # C#:
    #   int orbYTop = PriceTop + (int)((priceMax - orbHigh) / priceRange * PriceH);
    #   int orbYBot = PriceTop + (int)((priceMax - orbLow) / priceRange * PriceH);
    #   using (var orbBrush = new SolidBrush(BoxFill))
    #       g.FillRectangle(orbBrush, LeftPad, orbYTop, usableW, orbYBot - orbYTop);
    orb_y_top = _price_to_y(orb_high, price_min, price_range)
    orb_y_bot = _price_to_y(orb_low, price_min, price_range)

    if orb_y_bot > orb_y_top:
        draw_overlay.rectangle(
            [LEFT_PAD, orb_y_top, LEFT_PAD + usable_w - 1, orb_y_bot],
            fill=box_fill,
        )

    # ── Step 7: Draw range border lines ────────────────────────────────
    # C#:
    #   using (var orbPen = new Pen(BoxBorder, dashStyle))
    #   {
    #       g.DrawLine(orbPen, LeftPad, orbYTop, LeftPad + usableW, orbYTop);
    #       g.DrawLine(orbPen, LeftPad, orbYBot, LeftPad + usableW, orbYBot);
    #   }
    if box_dashed:
        # Dashed lines drawn directly on main image (not overlay) so the
        # gaps remain transparent rather than blending with the dark bg.
        _draw_dashed_hline(draw_main, LEFT_PAD, LEFT_PAD + usable_w, orb_y_top, box_border)
        _draw_dashed_hline(draw_main, LEFT_PAD, LEFT_PAD + usable_w, orb_y_bot, box_border)
    else:
        draw_overlay.line(
            [(LEFT_PAD, orb_y_top), (LEFT_PAD + usable_w, orb_y_top)],
            fill=box_border,
            width=1,
        )
        draw_overlay.line(
            [(LEFT_PAD, orb_y_bot), (LEFT_PAD + usable_w, orb_y_bot)],
            fill=box_border,
            width=1,
        )

    # ── Step 8: Draw VWAP line ─────────────────────────────────────────
    # C#:
    #   if (vwap != null && vwap.Length == bars.Length)
    #   {
    #       using (var vwapPen = new Pen(VwapLine))
    #       {
    #           for (int i = 1; i < bars.Length; i++)
    #           {
    #               int x0 = LeftPad + (i - 1) * barW + barW / 2;
    #               int x1 = LeftPad + i * barW + barW / 2;
    #               int y0 = PriceTop + (int)((priceMax - vwap[i-1]) / priceRange * PriceH);
    #               int y1 = PriceTop + (int)((priceMax - vwap[i])   / priceRange * PriceH);
    #               g.DrawLine(vwapPen, x0, y0, x1, y1);
    #           }
    #       }
    #   }
    if vwap_values is not None and len(vwap_values) == n:
        for i in range(1, n):
            x0 = LEFT_PAD + (i - 1) * bar_w + bar_w // 2
            x1 = LEFT_PAD + i * bar_w + bar_w // 2
            y0 = _price_to_y(vwap_values[i - 1], price_min, price_range)
            y1 = _price_to_y(vwap_values[i], price_min, price_range)
            draw_main.line([(x0, y0), (x1, y1)], fill=VWAP_LINE, width=1)

    # ── Step 9: Draw volume bars ───────────────────────────────────────
    # C#:
    #   for (int i = 0; i < bars.Length; i++)
    #   {
    #       int x = LeftPad + i * barW;
    #       int vh = (int)(bars[i].Volume / volMax * VolPanelH);
    #       int vy = H - vh;
    #       Color vc = bars[i].Close >= bars[i].Open ? VolBull : VolBear;
    #       using (var vb = new SolidBrush(vc))
    #           g.FillRectangle(vb, x, vy, bodyW, vh);
    #   }
    H - VOL_PANEL_H
    for i in range(n):
        x = LEFT_PAD + i * bar_w
        bar = bars[i]
        vh = _vol_to_h(bar.volume, vol_max)
        if vh <= 0:
            continue
        vy = H - vh
        vc = VOL_BULL if bar.is_bullish else VOL_BEAR
        draw_overlay.rectangle(
            [x, vy, x + body_w - 1, vy + vh - 1],
            fill=vc,
        )

    # ── Step 10: Draw candlesticks (bodies + wicks) ────────────────────
    # C#:
    #   for (int i = 0; i < bars.Length; i++)
    #   {
    #       var b = bars[i];
    #       int x = LeftPad + i * barW;
    #       int xMid = x + barW / 2;
    #       bool bull = b.Close >= b.Open;
    #       Color cc = bull ? BullCandle : BearCandle;
    #
    #       // Wick
    #       int yHigh = PriceTop + (int)((priceMax - b.High) / priceRange * PriceH);
    #       int yLow  = PriceTop + (int)((priceMax - b.Low)  / priceRange * PriceH);
    #       using (var wp = new Pen(cc))
    #           g.DrawLine(wp, xMid, yHigh, xMid, yLow);
    #
    #       // Body
    #       double bodyTop = Math.Max(b.Open, b.Close);
    #       double bodyBot = Math.Min(b.Open, b.Close);
    #       int yBodyTop = PriceTop + (int)((priceMax - bodyTop) / priceRange * PriceH);
    #       int yBodyBot = PriceTop + (int)((priceMax - bodyBot) / priceRange * PriceH);
    #       int bodyH = Math.Max(1, yBodyBot - yBodyTop);
    #       using (var bb = new SolidBrush(cc))
    #           g.FillRectangle(bb, x + 1, yBodyTop, bodyW, bodyH);
    #   }
    for i in range(n):
        bar = bars[i]
        x = LEFT_PAD + i * bar_w
        x_mid = x + bar_w // 2
        is_bull = bar.is_bullish
        cc = BULL_CANDLE if is_bull else BEAR_CANDLE

        # Wick
        y_high = _price_to_y(bar.high, price_min, price_range)
        y_low = _price_to_y(bar.low, price_min, price_range)
        if y_low > y_high:
            draw_main.line([(x_mid, y_high), (x_mid, y_low)], fill=cc, width=1)

        # Body
        body_top_price = max(bar.open, bar.close)
        body_bot_price = min(bar.open, bar.close)
        y_body_top = _price_to_y(body_top_price, price_min, price_range)
        y_body_bot = _price_to_y(body_bot_price, price_min, price_range)
        body_h = max(1, y_body_bot - y_body_top)

        draw_main.rectangle(
            [x + 1, y_body_top, x + 1 + body_w - 1, y_body_top + body_h - 1],
            fill=cc,
        )

    # ── Step 11: Composite overlay onto main image ─────────────────────
    # Convert main image to RGBA for compositing, then back to RGB
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    img_final = img_rgba.convert("RGB")

    return img_final


# ---------------------------------------------------------------------------
# File-saving helpers
# ---------------------------------------------------------------------------


def render_parity_to_file(
    bars: Sequence[ParityBar],
    orb_high: float,
    orb_low: float,
    vwap_values: Sequence[float] | None = None,
    direction: str | None = None,
    save_path: str = "",
    breakout_type: str | None = None,
) -> str | None:
    """Render and save a parity snapshot to a specific file path.

    Args:
        bars: Sequence of ParityBar (OHLCV).
        orb_high: Opening range high.
        orb_low: Opening range low.
        vwap_values: Per-bar VWAP values.
        direction: "long" or "short".
        save_path: Where to write the PNG.
        breakout_type: Box style selector — ``BreakoutType`` name or
                       ``box_style`` token.  Defaults to ORB gold-dashed.

    Returns:
        Path to the saved file, or None on failure.
    """
    img = render_parity_snapshot(bars, orb_high, orb_low, vwap_values, direction, breakout_type)
    if img is None:
        return None

    try:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        img.save(save_path, format="PNG")
        return save_path
    except Exception as exc:
        logger.error("Failed to save parity snapshot to %s: %s", save_path, exc)
        return None


def render_parity_to_temp(
    bars: Sequence[ParityBar],
    orb_high: float,
    orb_low: float,
    vwap_values: Sequence[float] | None = None,
    direction: str | None = None,
    temp_dir: str | None = None,
    label: str = "parity",
    breakout_type: str | None = None,
) -> str | None:
    """Render a parity snapshot to a temporary file.

    Mirrors C# ``OrbChartRenderer.RenderToTemp()`` behavior.

    Args:
        bars: Sequence of ParityBar (OHLCV).
        orb_high: Opening range high.
        orb_low: Opening range low.
        vwap_values: Per-bar VWAP values.
        direction: "long" or "short".
        temp_dir: Directory for temp files (uses system temp if None).
        label: Filename label prefix.
        breakout_type: Box style selector — ``BreakoutType`` name or
                       ``box_style`` token.  Defaults to ORB gold-dashed.

    Returns:
        Path to the saved temp PNG, or None on failure.
    """
    img = render_parity_snapshot(bars, orb_high, orb_low, vwap_values, direction, breakout_type)
    if img is None:
        return None

    try:
        dir_path = temp_dir or tempfile.gettempdir()
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{label}.png")
        img.save(path, format="PNG")
        return path
    except Exception as exc:
        logger.error("Failed to save temp parity snapshot: %s", exc)
        return None


# ---------------------------------------------------------------------------
# DataFrame adapter — converts pandas OHLCV to ParityBar list
# ---------------------------------------------------------------------------


def dataframe_to_parity_bars(
    df: Any,
    max_bars: int = 0,
) -> list[ParityBar]:
    """Convert a pandas DataFrame with OHLCV columns to a list of ParityBar.

    Accepts column names in any common capitalisation:
    Open/open, High/high, Low/low, Close/close, Volume/volume.

    Args:
        df: pandas DataFrame with OHLCV data.
        max_bars: If > 0, take only the last ``max_bars`` rows.

    Returns:
        List of ParityBar objects.
    """
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    # Normalise column names
    col_map: dict[str, str] = {}
    for col in df.columns:
        low = str(col).lower()
        if low == "open":
            col_map["open"] = col
        elif low == "high":
            col_map["high"] = col
        elif low == "low":
            col_map["low"] = col
        elif low == "close":
            col_map["close"] = col
        elif low == "volume":
            col_map["volume"] = col

    required = {"open", "high", "low", "close"}
    if not required.issubset(col_map.keys()):
        logger.warning(
            "DataFrame missing required OHLC columns: %s (have: %s)",
            required - col_map.keys(),
            list(df.columns),
        )
        return []

    if max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:]

    bars: list[ParityBar] = []
    has_volume = "volume" in col_map

    for _, row in df.iterrows():
        try:
            bars.append(
                ParityBar(
                    open=float(row[col_map["open"]]),
                    high=float(row[col_map["high"]]),
                    low=float(row[col_map["low"]]),
                    close=float(row[col_map["close"]]),
                    volume=float(row[col_map["volume"]]) if has_volume else 0.0,
                )
            )
        except (ValueError, TypeError):
            continue

    return bars


def compute_vwap_from_bars(bars: Sequence[ParityBar]) -> list[float]:
    """Compute a running VWAP series from ParityBar data.

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    where typical_price = (high + low + close) / 3.

    Returns a list of floats the same length as bars. If a bar has
    zero cumulative volume, the VWAP defaults to the bar's close.
    """
    cum_tp_vol = 0.0
    cum_vol = 0.0
    vwap_out: list[float] = []

    for bar in bars:
        tp = (bar.high + bar.low + bar.close) / 3.0
        cum_tp_vol += tp * bar.volume
        cum_vol += bar.volume
        if cum_vol > 0:
            vwap_out.append(cum_tp_vol / cum_vol)
        else:
            vwap_out.append(bar.close)

    return vwap_out


# ---------------------------------------------------------------------------
# Batch rendering for dataset generation
# ---------------------------------------------------------------------------


def render_parity_batch(
    bars_df: Any,
    orb_high: float,
    orb_low: float,
    direction: str | None = None,
    save_path: str = "",
    max_bars: int = 60,
    compute_vwap: bool = True,
    breakout_type: str | None = None,
) -> str | None:
    """Convenience function: DataFrame → parity PNG file.

    Combines ``dataframe_to_parity_bars()``, optional VWAP computation,
    and ``render_parity_to_file()`` into a single call suitable for
    the dataset generator.

    Args:
        bars_df: pandas DataFrame with OHLCV columns.
        orb_high: Opening range high.
        orb_low: Opening range low.
        direction: "long" or "short".
        save_path: Output PNG path.
        max_bars: Maximum bars to include (tail).
        compute_vwap: If True, compute VWAP from bar data.
        breakout_type: Box style selector — ``BreakoutType`` name or
                       ``box_style`` token.  Defaults to ORB gold-dashed.

    Returns:
        Path to saved PNG, or None on failure.
    """
    bars = dataframe_to_parity_bars(bars_df, max_bars=max_bars)
    if not bars:
        return None

    vwap = compute_vwap_from_bars(bars) if compute_vwap else None

    return render_parity_to_file(
        bars=bars,
        orb_high=orb_high,
        orb_low=orb_low,
        vwap_values=vwap,
        direction=direction,
        save_path=save_path,
        breakout_type=breakout_type,
    )


# ---------------------------------------------------------------------------
# Validation: compare Python render vs C# reference
# ---------------------------------------------------------------------------


def compare_with_reference(
    python_img: Image.Image | str,
    reference_img: Image.Image | str,
    tolerance: int = 3,
) -> dict[str, Any]:
    """Compare a Python-rendered parity image against a C# reference.

    Computes per-pixel absolute differences and reports statistics.
    Useful for regression testing renderer parity.

    Args:
        python_img: PIL Image or path to Python-rendered PNG.
        reference_img: PIL Image or path to C#-rendered reference PNG.
        tolerance: Maximum per-channel pixel difference considered a match.

    Returns:
        Dict with keys: match (bool), max_diff, mean_diff,
        mismatch_pixels, mismatch_pct, total_pixels.
    """
    if not _PIL_AVAILABLE:
        return {"error": "Pillow not available"}

    if isinstance(python_img, str):
        python_img = Image.open(python_img).convert("RGB")
    if isinstance(reference_img, str):
        reference_img = Image.open(reference_img).convert("RGB")

    # Ensure same size
    if python_img.size != reference_img.size:
        return {
            "match": False,
            "error": f"Size mismatch: {python_img.size} vs {reference_img.size}",
        }

    py_arr = np.array(python_img, dtype=np.int16)
    ref_arr = np.array(reference_img, dtype=np.int16)

    diff = np.abs(py_arr - ref_arr)
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())
    total_pixels = py_arr.shape[0] * py_arr.shape[1]
    # A pixel mismatches if ANY channel exceeds tolerance
    mismatch_mask = diff.max(axis=2) > tolerance
    mismatch_pixels = int(mismatch_mask.sum())
    mismatch_pct = mismatch_pixels / total_pixels * 100 if total_pixels > 0 else 0.0

    return {
        "match": max_diff <= tolerance,
        "max_diff": max_diff,
        "mean_diff": round(mean_diff, 3),
        "mismatch_pixels": mismatch_pixels,
        "mismatch_pct": round(mismatch_pct, 2),
        "total_pixels": total_pixels,
    }

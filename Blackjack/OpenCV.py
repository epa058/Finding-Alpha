"""
Blackjack table reader: card recognition, pile counting, tracking, Hi-Lo count.

Built for the Loto-Quebec / Espacejeux online blackjack table, and should work
on similar tables: green felt, white cards with rank+suit in the top-left
corner.

How it works (per frame)
------------------------
1. TABLE   Find the felt (the dominant saturated colour in the middle of
           the frame, so green, blue or red tables all work) and keep only
           what lies on it. This drops the balance bar, buttons and side
           panels without hard-coded crops.
2. BLOBS   White regions on the felt, lightly closed and hole-filled.
           One blob = one card or one overlapping pile (a hand).
3. INDEX   Inside each blob, pull the dark/red "ink" connected components
           and look for the card index: a rank glyph with a suit glyph just
           below it, same colour. A fanned pile keeps every card's top-left
           index visible, so the number of indices in a blob is the number of
           cards in the pile, whichever way the pile is fanned.
           Glyphs are cropped to their bounding box and resized to a fixed
           size before comparison with templates/, so recognition doesn't
           depend on screen resolution or browser zoom. (This is the
           "isolate rank/suit, resize, compare" idea from Cards.py in
           OpenCV-Playing-Card-Detector, adapted for screen captures.)
4. TRACK   Detections are matched to tracks frame to frame (Hungarian
           assignment on predicted position plus a label check). A card only
           counts once it has been seen for a few frames AND has stopped
           moving, so cards sliding in from the shoe, piles being moved and
           cards swept away at the end of a round aren't double-counted.
           Tracks that disappear briefly are re-linked by rank+suit.
5. COUNT   Hi-Lo running count, true count (by decks remaining), per-hand
           totals, cards per pile.

Usage
-----
    python OpenCV.py                         # analyse screenshot2.png
    python OpenCV.py screenshot.png          # analyse another image
    python OpenCV.py --video capture.mp4     # track + count through a recording
    python OpenCV.py --screen                # live, whole primary monitor
    python OpenCV.py --screen --region 0,0,1920,1080
    python OpenCV.py --simulate screenshot2.png   # tracking self-test (moves the cards around)

Live keys: q = quit, r = new shoe (reset count), p = pause, s = save frame.
Requires: opencv-python, numpy. Optional: scipy (better matching), mss (fast screen grab).
"""

import argparse
import hashlib
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # greedy fallback is fine for a handful of cards
    linear_sum_assignment = None


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
SUIT_COLOR = {'Hearts': 'red', 'Diamonds': 'red', 'Clubs': 'black', 'Spades': 'black'}
SUIT_LETTER = {'Hearts': 'H', 'Diamonds': 'D', 'Clubs': 'C', 'Spades': 'S'}

HILO = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
        '7': 0, '8': 0, '9': 0,
        '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1}
BJ_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}

# Card geometry in units of the rank-glyph height (measured on screenshot2:
# glyph 30 px, card 148x200, glyph top-left 8,6 px from the card corner).
CARD_W_PER_GLYPH = 4.93
CARD_H_PER_GLYPH = 6.67
GLYPH_OFF_X = 0.27
GLYPH_OFF_Y = 0.20


@dataclass
class Config:
    # colour thresholds (OpenCV HSV: H 0-180)
    felt_s_min: int = 150          # felt hue is found automatically (dominant saturated hue)
    felt_hue_tol: int = 12
    felt_v: tuple = (25, 200)
    white_s_max: int = 45
    white_v_min: int = 200
    ink_thresh: int = 110          # 255 - min(B,G,R); red and black both come out high
    red_thresh: int = 70           # R - max(G,B) to call a glyph red
    # recognition
    rank_min_score: float = 0.55
    suit_min_score: float = 0.55
    index_min_score: float = 0.50  # combined rank*suit score
    # tracking
    min_hits: int = 3              # frames a card must be seen before it counts
    settle_frames: int = 3         # ... and stationary over this many frames
    settle_tol: float = 0.06       # "stationary" = moved < this * card height
    max_missed: int = 20           # frames a track can go unseen before it is dropped
    relink_frames: int = 60        # lost tracks can be re-linked by label this long
    clear_frames: int = 12         # empty-table frames that end a round
    gate_same: float = 3.0         # max jump (card heights) for a same-label match
    gate_diff: float = 0.20        # max jump for a match whose label disagrees


# ════════════════════════════════════════════════════════════════════════════
#  GLYPH TEMPLATES
# ════════════════════════════════════════════════════════════════════════════

def ink_map(bgr):
    """Ink intensity: 0 on white paper, ~255 on black or red print."""
    return 255 - bgr.min(axis=2)


def red_map(bgr):
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    return np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)


def tight_crop(ink, thresh=60):
    """Crop an ink image to the bounding box of its visible ink."""
    ys, xs = np.where(ink > thresh)
    if len(xs) == 0:
        return None
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalise_glyph(ink_crop, size, pad=3):
    """Resize a tight glyph crop to a fixed size and pad it (lets templates slide a few px)."""
    g = cv2.resize(ink_crop.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


RANK_NORM = (28, 40)            # (w, h) rank glyphs are normalised to
SUIT_NORM = (32, 32)
TEMPLATE_SCALES = (0.90, 0.95, 1.0)


class TemplateBank:
    """
    A set of same-kind glyph templates (all ranks, or all suits) prepared for fast
    sliding normalised cross-correlation. The candidate glyph is normalised to a
    fixed size and padded; every template is slid over it at a few scales and
    the best NCC is kept. The few pixels of slack are what separate clubs from
    spades reliably (plain fixed-size correlation scores them almost the same).
    All templates of one scale go through a single matrix product per candidate.
    cv2.matchTemplate is ~100x slower on images this small.
    """

    def __init__(self, entries, size):
        # entries: [(name, color, tight_ink_crop)]
        self.size = size
        self.names = [e[0] for e in entries]
        self.colors = np.array([e[1] for e in entries])
        self.aspects = np.array([e[2].shape[1] / e[2].shape[0] for e in entries], np.float32)
        self.banks = []
        for s in TEMPLATE_SCALES:
            tw, th = int(size[0] * s), int(size[1] * s)
            mats = []
            for e in entries:
                t = cv2.resize(e[2].astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA).ravel()
                t = t - t.mean()
                mats.append(t / (np.linalg.norm(t) + 1e-6))
            self.banks.append(((th, tw), np.array(mats, np.float32)))

    def scores(self, ink_crop):
        norm = normalise_glyph(ink_crop, self.size)
        best = np.full(len(self.names), -1.0, np.float32)
        for (th, tw), mat in self.banks:
            win = np.lib.stride_tricks.sliding_window_view(norm, (th, tw)).reshape(-1, th * tw)
            win = win - win.mean(axis=1, keepdims=True)
            win /= (np.linalg.norm(win, axis=1, keepdims=True) + 1e-6)
            best = np.maximum(best, (win @ mat.T).max(axis=0))
        aspect = ink_crop.shape[1] / float(ink_crop.shape[0])
        return best - 0.35 * np.abs(np.log(aspect / self.aspects))


class GlyphLibrary:
    """
    Loads templates/rank_<R>_<color>.png and templates/suit_<suit>_<color>.png
    (rank_9.png with no colour also works). Each template is cropped to its ink
    bounding box so it can be compared with glyphs at any screen scale.
    """

    def __init__(self, folder, cfg):
        self.cfg = cfg
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Template folder '{folder}' not found")
        ranks, suits = [], []
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            parts = os.path.splitext(fn)[0].split('_')
            if len(parts) < 2 or parts[0] not in ('rank', 'suit'):
                continue
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img = self._to_bgr_on_white(img)
            ink = ink_map(img)
            ys, xs = np.where(ink > 60)
            if len(xs) == 0:
                continue
            sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            crop, bgr_crop = ink[sl], img[sl]
            is_red = np.mean(red_map(bgr_crop)[crop > cfg.ink_thresh] > cfg.red_thresh) > 0.5
            color = 'red' if is_red else 'black'
            if parts[0] == 'rank' and parts[1].upper() in RANKS:
                ranks.append((parts[1].upper(), color, crop))
            elif parts[0] == 'suit' and parts[1].capitalize() in SUIT_COLOR:
                name = parts[1].capitalize()
                suits.append((name, SUIT_COLOR[name], crop))
        for kind, have, full in (('ranks', {r[0] for r in ranks}, RANKS), ('suits', {s[0] for s in suits}, SUITS)):
            missing = sorted(set(full) - have)
            if missing:
                print(f"[WARN] no templates for {kind}: {missing}")
        if not ranks or not suits:
            raise RuntimeError(f"No usable rank/suit templates in '{folder}'")
        self.ranks = TemplateBank(ranks, RANK_NORM)
        self.suits = TemplateBank(suits, SUIT_NORM)
        print(f"Loaded {len(ranks)} rank and {len(suits)} suit templates from '{folder}'")

    @staticmethod
    def _to_bgr_on_white(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            return (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        return img

    @staticmethod
    def _pick(bank, scores, allowed):
        scores = np.where(allowed, scores, -9.0)
        i = int(np.argmax(scores))
        return bank.names[i], float(scores[i])

    def classify_rank(self, ink_crop, color):
        """Shape decides the rank. Prefer same-colour templates, fall back to the other colour."""
        s = self.ranks.scores(ink_crop)
        same = self.ranks.colors == color
        covered = {n for n, ok in zip(self.ranks.names, same) if ok}
        allowed = same | np.array([n not in covered for n in self.ranks.names])
        return self._pick(self.ranks, s, allowed)

    def classify_suit(self, ink_crop, color):
        return self._pick(self.suits, self.suits.scores(ink_crop), self.suits.colors == color)


# ════════════════════════════════════════════════════════════════════════════
#  DETECTION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CardDetection:
    rank: str
    suit: str
    score: float
    index_box: tuple          # (x, y, w, h) of rank+suit, frame coords
    glyph_h: float            # rank glyph height in px (sets the card scale)
    blob_id: int = -1
    role: str = 'player'      # 'dealer' / 'player'
    hand: int = -1            # hand number (a blob can hold two hands after a split)

    @property
    def label(self):
        return f"{self.rank}{SUIT_LETTER[self.suit]}"

    @property
    def anchor(self):         # rank glyph top-left: the tracked point
        return np.array(self.index_box[:2], dtype=np.float32)

    @property
    def card_h(self):
        return self.glyph_h * CARD_H_PER_GLYPH

    @property
    def card_box(self):
        """Full card rectangle estimated from the index (back cards are partly hidden)."""
        x, y = self.index_box[:2]
        gh = self.glyph_h
        return (int(x - GLYPH_OFF_X * gh), int(y - GLYPH_OFF_Y * gh),
                int(CARD_W_PER_GLYPH * gh), int(CARD_H_PER_GLYPH * gh))


@dataclass
class Blob:
    bid: int
    bbox: tuple               # x, y, w, h
    mask: np.ndarray          # filled mask cropped to bbox
    role: str = 'player'
    cards: list = field(default_factory=list)
    hands: list = field(default_factory=list)   # [[CardDetection, ...], ...]


@dataclass
class FrameResult:
    table_rect: tuple
    blobs: list
    detections: list


class TableReader:
    """Stateless per-frame detector: frame -> card detections grouped into piles."""

    def __init__(self, template_folder='templates', cfg=None):
        self.cfg = cfg or Config()
        self.lib = GlyphLibrary(template_folder, self.cfg)
        self.k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._cache = OrderedDict()

    # ── 1. table ────────────────────────────────────────────────────────────
    def find_table(self, frame):
        """Bounding rect + filled mask of the felt (largest region of the dominant saturated colour)."""
        c = self.cfg
        ih, iw = frame.shape[:2]
        f = 0.25
        small = cv2.resize(frame, (max(1, int(iw * f)), max(1, int(ih * f))), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        sat_ok = (sat >= c.felt_s_min) & (val >= c.felt_v[0]) & (val <= c.felt_v[1])
        sh, sw = hue.shape
        centre = np.zeros_like(sat_ok)
        centre[sh // 4:3 * sh // 4, sw // 4:3 * sw // 4] = True
        hist = np.bincount(hue[sat_ok & centre].ravel(), minlength=180).astype(np.float32)
        if hist.sum() < 50:
            return (0, 0, iw, ih), np.full((ih, iw), 255, np.uint8)   # no felt: use everything
        hist = np.convolve(np.r_[hist[-5:], hist, hist[:5]], np.ones(5), 'same')[5:-5]   # hue is circular
        h0 = int(np.argmax(hist))
        dh = np.abs(((hue.astype(np.int16) - h0 + 90) % 180) - 90)
        felt = ((dh <= c.felt_hue_tol) & sat_ok).astype(np.uint8) * 255
        felt = cv2.morphologyEx(felt, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 0.05 * small.size / 3:
            return (0, 0, iw, ih), np.full((ih, iw), 255, np.uint8)   # no felt: use everything
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea)) / f
        mask = np.zeros((ih, iw), np.uint8)
        cv2.fillPoly(mask, [hull.astype(np.int32)], 255)
        x, y, w, h = cv2.boundingRect(hull.astype(np.int32))
        x, y = max(0, x), max(0, y)
        return (x, y, min(w, iw - x), min(h, ih - y)), mask

    # ── 2. blobs ────────────────────────────────────────────────────────────
    def find_blobs(self, frame, table_mask, table_rect):
        c = self.cfg
        tx, ty, tw, th = table_rect
        roi = frame[ty:ty + th, tx:tx + tw]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, c.white_v_min), (180, c.white_s_max, 255))
        white = cv2.bitwise_and(white, table_mask[ty:ty + th, tx:tx + tw])
        # Close just enough to bridge the card-edge lines inside a pile.
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(tx, ty))
        min_area = (0.045 * table_rect[2]) ** 2      # ~1/4 of a card (card width ~0.086 x table width); drops chips
        blobs = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [cnt - [x, y]], -1, 255, -1)       # filled => holes gone
            blobs.append(Blob(len(blobs), (x, y, w, h), m))
        return blobs

    # ── 3. index detection inside a blob ────────────────────────────────────
    def _candidates(self, bgr, mask):
        """Ink connected components (+ merged '10' pairs, + split rank/suit blobs)."""
        c = self.cfg
        inner = cv2.erode(mask, self.k3, iterations=2)          # drop felt fringe at card edges
        ink = ink_map(bgr)
        bw = ((ink > c.ink_thresh) & (inner > 0)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        comps = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if h < 7 or w < 2 or a < 12:
                continue
            fill = a / float(w * h)
            if fill < 0.12 or h > 0.5 * mask.shape[0] or w > 0.5 * max(mask.shape[1], 40) and h < 0.3 * w:
                continue   # frame lines, card-edge shadows
            comps.append({'x': x, 'y': y, 'w': w, 'h': h, 'ids': [i]})

        cands = list(comps)
        # '10': a thin '1' followed closely by a same-height '0'
        for a_ in comps:
            if a_['w'] / a_['h'] > 0.5:
                continue
            for b_ in comps:
                if b_ is a_ or b_['x'] <= a_['x']:
                    continue
                gap = b_['x'] - (a_['x'] + a_['w'])
                ov = min(a_['y'] + a_['h'], b_['y'] + b_['h']) - max(a_['y'], b_['y'])
                if -2 <= gap <= 0.4 * a_['h'] and ov > 0.75 * min(a_['h'], b_['h']) \
                        and 0.75 < b_['h'] / a_['h'] < 1.33:
                    x0, y0 = min(a_['x'], b_['x']), min(a_['y'], b_['y'])
                    x1 = max(a_['x'] + a_['w'], b_['x'] + b_['w'])
                    y1 = max(a_['y'] + a_['h'], b_['y'] + b_['h'])
                    cands.append({'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0, 'ids': a_['ids'] + b_['ids']})
        # rank and suit touching (happens when the table is rendered small)
        for a_ in comps:
            if a_['h'] / a_['w'] < 1.9:
                continue
            sub = (lab[a_['y']:a_['y'] + a_['h'], a_['x']:a_['x'] + a_['w']] == a_['ids'][0])
            rows = sub.sum(axis=1)
            lo, hi = int(0.35 * a_['h']), int(0.7 * a_['h'])
            if hi <= lo:
                continue
            cut = lo + int(np.argmin(rows[lo:hi]))
            if rows[cut] <= max(1, 0.15 * a_['w']):
                for (yy, hh) in ((a_['y'], cut), (a_['y'] + cut + 1, a_['h'] - cut - 1)):
                    part = sub[yy - a_['y']:yy - a_['y'] + hh]
                    cols = np.where(part.any(axis=0))[0]
                    rws = np.where(part.any(axis=1))[0]
                    if len(cols) and len(rws) and len(rws) >= 5:
                        cands.append({'x': a_['x'] + cols[0], 'y': yy + rws[0], 'w': cols[-1] - cols[0] + 1,
                                      'h': rws[-1] - rws[0] + 1, 'ids': a_['ids'], 'clip': True})
        return cands, lab, ink

    def _describe(self, cand, lab, ink, redm):
        c = self.cfg
        x, y, w, h = cand['x'], cand['y'], cand['w'], cand['h']
        sel = np.isin(lab[y:y + h, x:x + w], cand['ids'])
        on = sel & (ink[y:y + h, x:x + w] > c.ink_thresh)
        if on.sum() < 8:
            return None
        sel = cv2.dilate(sel.astype(np.uint8), self.k3) > 0     # keep anti-aliased edges
        crop = tight_crop(np.where(sel, ink[y:y + h, x:x + w], 0))
        if crop is None or min(crop.shape) < 3:
            return None
        cand['color'] = 'red' if np.mean(redm[y:y + h, x:x + w][on] > c.red_thresh) > 0.5 else 'black'
        cand['crop'] = crop
        cand['aspect'] = crop.shape[1] / float(crop.shape[0])
        return cand

    @staticmethod
    def _index_geometry(rc, sc):
        """Is sc where the suit of an upright index would sit below rank glyph rc?"""
        if sc is rc or sc['color'] != rc['color']:
            return False
        if set(sc['ids']) & set(rc['ids']) and not (sc.get('clip') and rc.get('clip')):
            return False
        gap = sc['y'] - (rc['y'] + rc['h'])
        dx = abs((sc['x'] + sc['w'] / 2) - (rc['x'] + rc['w'] / 2))
        hr = sc['h'] / rc['h']
        return -0.12 * rc['h'] <= gap <= 0.45 * rc['h'] and dx <= 0.35 * rc['h'] and 0.5 <= hr <= 1.15

    def find_indices(self, frame, blob):
        """All upright rank+suit indices inside one blob (= the cards of one pile)."""
        c = self.cfg
        bx, by, bw_, bh_ = blob.bbox
        bgr = frame[by:by + bh_, bx:bx + bw_]
        cands, lab, ink = self._candidates(bgr, blob.mask)
        redm = red_map(bgr)
        cands = [cd for cd in cands if self._describe(cd, lab, ink, redm) is not None]
        rank_like = [cd for cd in cands if 0.25 < cd['aspect'] < 1.35]
        suit_like = [cd for cd in cands if 0.55 < cd['aspect'] < 1.5]

        # Geometry first (cheap): a rank glyph with a same-colour glyph right under it.
        # Pips are too far apart to qualify, so only real indices get template-matched.
        geo = [(rc, sc) for rc in rank_like for sc in suit_like if self._index_geometry(rc, sc)]
        rank_res, suit_res = {}, {}
        pairs = []
        for rc, sc in geo:
            if id(rc) not in rank_res:
                rank_res[id(rc)] = self.lib.classify_rank(rc['crop'], rc['color'])
            if id(sc) not in suit_res:
                suit_res[id(sc)] = self.lib.classify_suit(sc['crop'], sc['color'])
            (r, rs), (s, ss) = rank_res[id(rc)], suit_res[id(sc)]
            if rs < c.rank_min_score or ss < c.suit_min_score:
                continue
            gap = sc['y'] - (rc['y'] + rc['h'])
            score = rs * ss - 0.3 * max(0.0, gap / rc['h'] - 0.2)
            if score >= c.index_min_score:
                pairs.append((score, rc, r, sc, s))

        # non-maximum suppression: every glyph belongs to at most one index
        pairs.sort(key=lambda p: -p[0])
        used, out = set(), []
        for score, rc, r, sc, s in pairs:
            ids = set(rc['ids']) | set(sc['ids'])
            if ids & used:
                continue
            used |= ids
            x0 = min(rc['x'], sc['x'])
            x1 = max(rc['x'] + rc['w'], sc['x'] + sc['w'])
            out.append(CardDetection(r, s, score, (bx + x0, by + rc['y'], x1 - x0, sc['y'] + sc['h'] - rc['y']),
                                     float(rc['h']), blob.bid, blob.role))
        return out

    # ── full frame ──────────────────────────────────────────────────────────
    def process(self, frame):
        table_rect, table_mask = self.find_table(frame)
        blobs = self.find_blobs(frame, table_mask, table_rect)
        tx, ty, tw, th = table_rect
        dets = []
        for b in blobs:
            cy = b.bbox[1] + b.bbox[3] / 2
            b.role = 'dealer' if cy < ty + 0.40 * th else 'player'
            b.cards = self._cached_indices(frame, b)
            dets.extend(b.cards)
        # every index on one table is the same size: drop outliers (stray text, UI glyphs)
        if len(dets) >= 2:
            med = np.median([d.glyph_h for d in dets])
            keep = {id(d) for d in dets if 0.75 * med <= d.glyph_h <= 1.3 * med}
            for b in blobs:
                b.cards = [d for d in b.cards if id(d) in keep]
            dets = [d for d in dets if id(d) in keep]
        blobs = [b for b in blobs if b.cards]
        hand_no = 0
        for b in sorted(blobs, key=lambda b: (b.role != 'dealer', b.bbox[0])):
            b.cards.sort(key=lambda d: d.index_box[0])        # dealing order: left -> right
            # A fanned hand steps ~0.25 card widths per card; a bigger jump means a
            # second hand touching this one (e.g. after a split).
            b.hands = []
            for d in b.cards:
                if not b.hands or d.index_box[0] - b.hands[-1][-1].index_box[0] > 0.6 * d.card_box[2]:
                    b.hands.append([])
                    hand_no += 1
                d.hand = hand_no
                b.hands[-1].append(d)
        return FrameResult(table_rect, blobs, dets)

    def _cached_indices(self, frame, blob):
        """
        Most live frames are identical to the last one, and a pile that slides
        keeps the same pixels. Cache results by blob content so only blobs that
        changed are re-analysed; a moved blob reuses them, shifted.
        """
        bx, by, bw_, bh_ = blob.bbox
        key = hashlib.blake2b(np.ascontiguousarray(frame[by:by + bh_, bx:bx + bw_]).data, digest_size=16)
        key.update(blob.mask.data)
        key = key.digest()
        hit = self._cache.get(key)
        if hit is None:
            dets = self.find_indices(frame, blob)
            hit = [(d.rank, d.suit, d.score, (d.index_box[0] - bx, d.index_box[1] - by) + tuple(d.index_box[2:]),
                    d.glyph_h) for d in dets]
            self._cache[key] = hit
            if len(self._cache) > 256:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return [CardDetection(r, s, sc, (ix + bx, iy + by, iw, ih), gh, blob.bid, blob.role)
                for r, s, sc, (ix, iy, iw, ih), gh in hit]


# ════════════════════════════════════════════════════════════════════════════
#  HAND HELPERS
# ════════════════════════════════════════════════════════════════════════════

def hand_value(ranks):
    """Blackjack total and whether it is soft."""
    total = sum(BJ_VALUE[r] for r in ranks)
    aces = sum(r == 'A' for r in ranks)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0


def as_strategy_cards(labels):
    """[('8','Spades'), ...] -> [{'rank': '8', 'suit': 'Spades'}] as used by BasicStrategy.py."""
    return [{'rank': r, 'suit': s} for r, s in labels]


# ════════════════════════════════════════════════════════════════════════════
#  TRACKING + COUNTING
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Track:
    tid: int
    anchor: np.ndarray
    glyph_h: float
    first_frame: int
    last_frame: int
    votes: dict = field(default_factory=dict)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    history: list = field(default_factory=list)       # recent anchors (for "has it stopped?")
    hits: int = 0
    misses: int = 0
    counted_as: object = None                           # (rank, suit) once counted
    role: str = 'player'
    hand: int = -1
    det: object = None                                  # latest detection

    @property
    def rank_suit(self):
        return max(self.votes.items(), key=lambda kv: kv[1])[0]

    @property
    def label(self):
        r, s = self.rank_suit
        return f"{r}{SUIT_LETTER[s]}"

    @property
    def card_h(self):
        return self.glyph_h * CARD_H_PER_GLYPH

    def vote_share(self):
        tot = sum(self.votes.values())
        return self.votes[self.rank_suit] / tot if tot else 0.0


class CardTracker:
    """
    Frame-to-frame card tracking with a Hi-Lo counter that counts every physical
    card exactly once.

    * Matching: Hungarian assignment on distance from each track's predicted
      position (constant velocity), in card heights. A same-label match may
      jump up to cfg.gate_same card heights per frame (dealing animations are
      fast). A different-label match must be within cfg.gate_diff (and the
      track seen in the last couple of frames), which
      lets a misread card keep its identity without letting two neighbouring
      cards in a pile swap.
    * Re-linking: a detection with no match is attached to a recently lost
      track with the same rank+suit (card hidden by an animation, window
      dragged, etc.) instead of creating a new card.
    * Counting: a track counts once it has >= min_hits sightings, a stable
      majority label, and has stopped moving. If later frames outvote the
      label, the count is corrected rather than added again.
    * Rounds: when the table has been empty for cfg.clear_frames frames, the
      round is over and old tracks are dropped. The running count carries
      over until reset_shoe().
    """

    def __init__(self, num_decks=8, cfg=None):
        self.cfg = cfg or Config()
        self.num_decks = num_decks
        self.reset_shoe()

    def reset_shoe(self):
        self.tracks = []
        self.lost = []
        self.next_id = 1
        self.frame_idx = 0
        self.running_count = 0
        self.cards_seen = 0
        self.log = []                 # (frame, tid, rank, suit, hilo)
        self.round_no = 1
        self.empty_frames = 0

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def decks_remaining(self):
        return max(0.5, self.num_decks - self.cards_seen / 52.0)

    @property
    def true_count(self):
        return self.running_count / self.decks_remaining

    def update(self, detections):
        c = self.cfg
        self.frame_idx += 1
        pool = self.tracks + self.lost
        matches, un_det = self._associate(pool, detections)

        for t, d in matches:
            self._apply(t, d)

        matched_tracks = {id(t) for t, _ in matches}
        # re-link unmatched detections to lost tracks with the same identity
        still_new = []
        for d in un_det:
            cand = [t for t in pool if id(t) not in matched_tracks
                    and self.frame_idx - t.last_frame <= c.relink_frames
                    and (d.rank, d.suit) in t.votes and t.rank_suit == (d.rank, d.suit)]
            if cand:
                t = min(cand, key=lambda t: np.linalg.norm(t.anchor - d.anchor))
                t.history.clear()                       # it jumped: must settle again
                t.vel[:] = 0
                self._apply(t, d)
                matched_tracks.add(id(t))
            else:
                still_new.append(d)
        for d in still_new:
            t = Track(self.next_id, d.anchor.copy(), d.glyph_h, self.frame_idx, self.frame_idx)
            self.next_id += 1
            self._apply(t, d)
            pool.append(t)
            matched_tracks.add(id(t))

        # age out tracks that were not seen this frame
        self.tracks, self.lost = [], []
        for t in pool:
            if id(t) in matched_tracks:
                self.tracks.append(t)
                continue
            t.misses += 1
            t.anchor = t.anchor + t.vel
            t.vel *= 0.5
            if t.misses <= c.max_missed or (t.counted_as and self.frame_idx - t.last_frame <= c.relink_frames):
                self.lost.append(t)

        # count confirmed, settled cards
        for t in self.tracks:
            self._maybe_count(t)

        # round bookkeeping
        if detections:
            self.empty_frames = 0
        else:
            self.empty_frames += 1
            if self.empty_frames == c.clear_frames and (self.tracks or self.lost):
                self.lost = []
                self.round_no += 1
        return self.tracks

    # ── internals ───────────────────────────────────────────────────────────
    def _associate(self, pool, dets):
        c = self.cfg
        if not pool or not dets:
            return [], list(dets)
        INF = 1e6
        cost = np.full((len(pool), len(dets)), INF, np.float32)
        for i, t in enumerate(pool):
            pred = t.anchor + t.vel
            for j, d in enumerate(dets):
                dist = np.linalg.norm(pred - d.anchor) / max(t.card_h, 1.0)
                same = (d.rank, d.suit) == t.rank_suit
                if same and dist <= c.gate_same:
                    cost[i, j] = dist
                elif not same and dist <= c.gate_diff and t.misses <= 2:
                    cost[i, j] = dist + 1.0
        pairs = []
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            pairs = [(r, cl) for r, cl in zip(rows, cols) if cost[r, cl] < INF]
        else:
            used_r, used_c = set(), set()
            for flat in np.argsort(cost, axis=None):
                r, cl = divmod(int(flat), cost.shape[1])
                if cost[r, cl] >= INF:
                    break
                if r not in used_r and cl not in used_c:
                    pairs.append((r, cl))
                    used_r.add(r)
                    used_c.add(cl)
        matched_d = {cl for _, cl in pairs}
        return [(pool[r], dets[cl]) for r, cl in pairs], [d for j, d in enumerate(dets) if j not in matched_d]

    def _apply(self, t, d):
        if t.hits:
            step = d.anchor - t.anchor
            t.vel = 0.6 * step + 0.4 * t.vel
        t.anchor = d.anchor.copy()
        t.glyph_h = 0.8 * t.glyph_h + 0.2 * d.glyph_h if t.hits else d.glyph_h
        key = (d.rank, d.suit)
        t.votes[key] = t.votes.get(key, 0.0) + d.score
        t.hits += 1
        t.misses = 0
        t.last_frame = self.frame_idx
        t.role, t.hand, t.det = d.role, d.hand, d
        t.history.append(t.anchor.copy())
        if len(t.history) > 30:
            t.history.pop(0)

    def _settled(self, t):
        c = self.cfg
        if len(t.history) < c.settle_frames:
            return False
        recent = np.array(t.history[-c.settle_frames:])
        return np.max(np.linalg.norm(recent - recent[-1], axis=1)) <= c.settle_tol * t.card_h

    def _maybe_count(self, t):
        c = self.cfg
        if t.hits < c.min_hits or t.vote_share() < 0.6:
            return
        rs = t.rank_suit
        if t.counted_as is None:
            if not self._settled(t):
                return
            t.counted_as = rs
            self.running_count += HILO[rs[0]]
            self.cards_seen += 1
            self.log.append((self.frame_idx, t.tid, rs[0], rs[1], HILO[rs[0]]))
        elif t.counted_as != rs:                       # label corrected by later frames
            self.running_count += HILO[rs[0]] - HILO[t.counted_as[0]]
            self.log.append((self.frame_idx, t.tid, rs[0], rs[1], f"corrected from {t.counted_as[0]}"))
            t.counted_as = rs

    def hands(self):
        """Current hands on the table built from tracks: [(role, [track, ...]), ...]."""
        groups = {}
        for t in self.tracks:
            groups.setdefault(t.hand, []).append(t)
        out = []
        for _, ts in groups.items():
            ts.sort(key=lambda t: t.anchor[0])
            out.append((ts[0].role, ts))
        out.sort(key=lambda h: (h[0] != 'dealer', h[1][0].anchor[0]))
        return out


# ════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ════════════════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [(0, 220, 60), (255, 160, 0), (0, 180, 255), (230, 60, 230), (0, 220, 220), (80, 80, 255)]


def _text(img, txt, org, scale, color, thick=2, bg=(0, 0, 0)):
    (tw, th), bl = cv2.getTextSize(txt, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + bl + 1), bg, -1)
    cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_frame(frame, result, tracker=None, fps=None):
    out = frame.copy()
    s = max(0.5, frame.shape[1] / 2600)
    tx, ty, tw, th = result.table_rect
    cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (90, 90, 90), 1)

    # tracked label for each detection this frame (votes smooth out single-frame misreads)
    tracked = {}
    if tracker is not None:
        tracked = {id(t.det): t for t in tracker.tracks if t.last_frame == tracker.frame_idx}

    # hands / piles
    for b in result.blobs:
        for hand in b.hands:
            boxes = np.array([d.card_box for d in hand])
            x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
            x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
            col = (0, 200, 255) if b.role == 'dealer' else (255, 200, 0)
            cv2.rectangle(out, (int(x0) - 4, int(y0) - 4), (int(x1) + 4, int(y1) + 4), col, 2)
            ranks = [tracked[id(d)].rank_suit[0] if id(d) in tracked else d.rank for d in hand]
            total, soft = hand_value(ranks)
            tag = f"{'DEALER' if b.role == 'dealer' else 'PLAYER'}  {len(hand)} card{'s' if len(hand) > 1 else ''}" \
                  f"  = {'soft ' if soft else ''}{total}"
            _text(out, tag, (int(x0), int(y1 + 30 * s + 8)), 0.8 * s, col)

    # cards
    for d in result.detections:
        t = tracked.get(id(d))
        tid = t.tid if t else 0
        label = t.label if t else d.label
        col = PALETTE[tid % len(PALETTE)]
        cx, cy, cw, ch = d.card_box
        cv2.rectangle(out, (cx, cy), (cx + cw, cy + ch), col, 1)
        ix, iy, iw, ih = d.index_box
        cv2.rectangle(out, (ix - 2, iy - 2), (ix + iw + 2, iy + ih + 2), col, 2)
        txt = label + (f" #{tid}" if tid else "") + (" *" if t is not None and t.counted_as else "")
        _text(out, txt, (ix + iw + 8, iy + 20 * s), 0.75 * s, col)

    # HUD
    if tracker is not None:
        lines = [f"Running count: {tracker.running_count:+d}",
                 f"True count:    {tracker.true_count:+.2f}",
                 f"Cards seen:    {tracker.cards_seen}   decks left: {tracker.decks_remaining:.1f}",
                 f"Round:         {tracker.round_no}      (* = counted)"]
        if fps:
            lines.append(f"FPS:           {fps:.1f}")
        y0 = int(40 * s) + 10
        for i, ln in enumerate(lines):
            _text(out, ln, (15, y0 + i * int(38 * s)), 0.9 * s, (255, 255, 255), 2, (40, 40, 40))
    return out


# ════════════════════════════════════════════════════════════════════════════
#  RUNNERS
# ════════════════════════════════════════════════════════════════════════════

def describe_card(r, s):
    return f"{r} of {s}"


def run_image(path, reader, out_path='final_result.jpg', decks=8):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[ERROR] cannot read '{path}'")
    t0 = time.time()
    res = reader.process(img)
    dt = (time.time() - t0) * 1000
    print(f"\n{path}: {img.shape[1]}x{img.shape[0]}  table={tuple(int(v) for v in res.table_rect)}  ({dt:.0f} ms)")
    print("─" * 60)
    rc = 0
    for b in sorted(res.blobs, key=lambda b: (b.role != 'dealer', b.bbox[0])):
        for hand in b.hands:
            total, soft = hand_value([d.rank for d in hand])
            print(f"{b.role.upper():7s} hand at {tuple(int(v) for v in hand[0].card_box[:2])}: "
                  f"{len(hand)} card(s), total {'soft ' if soft else ''}{total}")
            for d in hand:
                rc += HILO[d.rank]
                print(f"   {describe_card(d.rank, d.suit):18s} score {d.score:.2f}  Hi-Lo {HILO[d.rank]:+d}  "
                      f"index at {tuple(int(v) for v in d.index_box[:2])}")
    print("─" * 60)
    print(f"Cards on table: {len(res.detections)}   Hi-Lo running count: {rc:+d}   "
          f"true count ({decks} decks): {rc / max(0.5, decks - len(res.detections) / 52):+.2f}")
    tracker = CardTracker(decks, reader.cfg)
    for _ in range(reader.cfg.min_hits + reader.cfg.settle_frames):   # a still image = a settled frame sequence
        tracker.update(res.detections)
    out = draw_frame(img, res, tracker)
    cv2.imwrite(out_path, out)
    print(f"Saved {out_path}")
    return res


def _loop(frames, reader, tracker, show=True, writer_path=None, max_width=1400):
    writer = None
    last, fps = time.time(), None
    paused = False
    for frame in frames:
        if frame is None:
            break
        res = reader.process(frame)
        tracker.update(res.detections)
        now = time.time()
        fps = 1.0 / max(now - last, 1e-6) if fps is None else 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        vis = draw_frame(frame, res, tracker, fps)
        if writer_path:
            if writer is None:
                writer = cv2.VideoWriter(writer_path, cv2.VideoWriter_fourcc(*'mp4v'), 15,
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if show:
            scale = min(1.0, max_width / vis.shape[1])
            cv2.imshow('Blackjack reader', cv2.resize(vis, None, fx=scale, fy=scale) if scale < 1 else vis)
            while True:
                k = cv2.waitKey(30 if paused else 1) & 0xFF
                if k == ord('q'):
                    if writer:
                        writer.release()
                    cv2.destroyAllWindows()
                    return
                if k == ord('r'):
                    tracker.reset_shoe()
                    print("[shoe reset]")
                if k == ord('s'):
                    cv2.imwrite(f"frame_{tracker.frame_idx:05d}.jpg", vis)
                if k == ord('p'):
                    paused = not paused
                if not paused:
                    break
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def _print_log(tracker):
    print("\nCount log (frame, track, rank, suit, Hi-Lo):")
    for row in tracker.log:
        print("  ", row)
    print(f"Final running count {tracker.running_count:+d}, true count {tracker.true_count:+.2f}, "
          f"cards seen {tracker.cards_seen}, rounds {tracker.round_no}")


def video_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] cannot open video '{path}'")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def screen_frames(region=None, monitor=1, fps_limit=15):
    try:
        import mss
        sct = mss.mss()
        mon = sct.monitors[monitor]
        box = {'left': mon['left'], 'top': mon['top'], 'width': mon['width'], 'height': mon['height']}
        if region:
            box = {'left': region[0], 'top': region[1], 'width': region[2], 'height': region[3]}
        grab = lambda: cv2.cvtColor(np.array(sct.grab(box)), cv2.COLOR_BGRA2BGR)
    except ImportError:
        from PIL import ImageGrab
        bbox = (region[0], region[1], region[0] + region[2], region[1] + region[3]) if region else None
        grab = lambda: cv2.cvtColor(np.array(ImageGrab.grab(bbox=bbox)), cv2.COLOR_RGB2BGR)
        print("[INFO] 'pip install mss' for faster screen capture")
    period = 1.0 / fps_limit
    while True:
        t0 = time.time()
        yield grab()
        time.sleep(max(0.0, period - (time.time() - t0)))


def simulate_frames(path, n=90):
    """
    Tracking self-test from a single screenshot: the table content slides
    around (like a window being dragged or cards animating), stops, moves
    again and briefly disappears. The count must not change after the first
    confirmation.
    """
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[ERROR] cannot read '{path}'")
    h, w = img.shape[:2]
    pad = cv2.copyMakeBorder(img, 200, 200, 300, 300, cv2.BORDER_REPLICATE)
    for i in range(n):
        if 55 <= i < 60:                   # cards hidden for 5 frames (plain felt)
            yield np.full((h, w, 3), (40, 70, 0), np.uint8)
            continue
        if i < 20:
            dx, dy = 0, 0
        elif i < 40:
            dx, dy = int(-12 * (i - 20)), int(5 * (i - 20))     # glide left/down 240x100 px
        elif i < 55:
            dx, dy = -240, 100
        else:
            dx, dy = 150, -120                                  # re-appears somewhere else
        yield pad[200 - dy:200 - dy + h, 300 - dx:300 - dx + w].copy()


def main():
    ap = argparse.ArgumentParser(description="Online blackjack card reader / tracker / Hi-Lo counter")
    ap.add_argument('image', nargs='?', default='screenshot2.png', help="screenshot to analyse")
    ap.add_argument('--video', help="video file to track through")
    ap.add_argument('--screen', action='store_true', help="live screen capture")
    ap.add_argument('--region', help="x,y,w,h screen region for --screen")
    ap.add_argument('--monitor', type=int, default=1)
    ap.add_argument('--simulate', action='store_true', help="tracking self-test on the image")
    ap.add_argument('--templates', default=None, help="template folder (default: ./templates)")
    ap.add_argument('--decks', type=int, default=8)
    ap.add_argument('--out', default=None, help="output image / video path")
    ap.add_argument('--no-show', action='store_true', help="don't open a window")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    tpl = args.templates or os.path.join(here, 'templates')
    if not os.path.exists(args.image) and os.path.exists(os.path.join(here, args.image)):
        args.image = os.path.join(here, args.image)
    reader = TableReader(tpl)
    tracker = CardTracker(args.decks, reader.cfg)

    if args.screen:
        region = tuple(int(v) for v in args.region.split(',')) if args.region else None
        _loop(screen_frames(region, args.monitor), reader, tracker, show=True, writer_path=args.out)
        _print_log(tracker)
    elif args.video:
        _loop(video_frames(args.video), reader, tracker, show=not args.no_show, writer_path=args.out)
        _print_log(tracker)
    elif args.simulate:
        _loop(simulate_frames(args.image), reader, tracker, show=not args.no_show,
              writer_path=args.out or 'simulation.mp4')
        _print_log(tracker)
    else:
        run_image(args.image, reader, args.out or 'final_result.jpg', args.decks)


if __name__ == '__main__':
    main()

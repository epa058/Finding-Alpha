"""
Tracking / counting regression test for OpenCV.py.

Builds a synthetic video of three blackjack rounds on the real table from
screenshot2.png: cards slide in from the shoe with motion blur, piles fan up
and to the right like the real game, the dealer's hole card is dealt face down
and flipped, a split moves a counted card, a pop-up briefly hides a pile, the
whole window is dragged mid-round, and every round is swept off the table.

The tracker must end with exactly the Hi-Lo count of the cards dealt, counting
each card once.

    python test_tracking.py            # prints the result, writes tracking_test.mp4
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import OpenCV as bj  # noqa: E402

CARD_W, CARD_H = 148, 200
PILE_STEP = np.array([38, -68])          # measured on screenshot2
SHOE = np.array([1900, 60])
DISCARD = np.array([300, 150])


# ── assets ───────────────────────────────────────────────────────────────────
def card_mask(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 45, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(m, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return cv2.dilate(m, np.ones((3, 3), np.uint8))


def load_assets():
    s2 = cv2.imread(os.path.join(HERE, 'screenshot2.png'))
    s1 = cv2.imread(os.path.join(HERE, 'screenshot.png'))
    red_base = s2[287:487, 1053:1201].copy()          # 8 of diamonds
    black_base = s1[287:487, 1053:1201].copy()        # 2 of clubs
    alpha = card_mask(red_base)

    # empty table: inpaint the cards, the orange total marker and the dealer total text
    bg = s2.copy()
    m = np.zeros(bg.shape[:2], np.uint8)
    m[280:495, 1045:1210] = 255
    m[700:1055, 1040:1275] = 255
    m[640:720, 1140:1270] = 255
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    m |= cv2.dilate(cv2.inRange(hsv, (5, 120, 150), (25, 255, 255)), np.ones((9, 9), np.uint8))
    m[370:420, 985:1025] = 255
    bg = cv2.inpaint(bg, m, 7, cv2.INPAINT_TELEA)
    return bg, red_base, black_base, alpha


def glyph(path, height):
    im = cv2.imread(path)
    ink = bj.ink_map(im)
    ys, xs = np.where(ink > 40)
    im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = height / im.shape[0]
    return cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def make_card(rank, suit, red_base, black_base):
    """A full card with the right index in both corners (pips come from the base card)."""
    red = bj.SUIT_COLOR[suit] == 'red'
    card = (red_base if red else black_base).copy()
    card[4:70, 3:34] = 255
    card[130:196, 114:145] = 255
    t = os.path.join(HERE, 'templates')
    rf = {'9': ('rank_9.png', 'rank_9_red.png')}.get(rank, (f'rank_{rank}_black.png', f'rank_{rank}_red.png'))
    rg = glyph(os.path.join(t, rf[1] if red else rf[0]), 30)
    sg = glyph(os.path.join(t, f"suit_{suit.lower()}_{'red' if red else 'black'}.png"), 25)
    rx = 8 if rank != '10' else 4
    ry = 6
    cx = rx + rg.shape[1] // 2
    sx, sy = max(2, cx - sg.shape[1] // 2), ry + rg.shape[0] + 2
    idx = np.full((sy + sg.shape[0] - ry, max(rx + rg.shape[1], sx + sg.shape[1]) - min(rx, sx), 3), 255, np.uint8)
    ox = min(rx, sx)
    idx[0:rg.shape[0], rx - ox:rx - ox + rg.shape[1]] = rg
    idx[sy - ry:sy - ry + sg.shape[0], sx - ox:sx - ox + sg.shape[1]] = sg
    card[ry:ry + idx.shape[0], ox:ox + idx.shape[1]] = np.minimum(card[ry:ry + idx.shape[0], ox:ox + idx.shape[1]], idx)
    rot = cv2.rotate(idx, cv2.ROTATE_180)
    bx, by = CARD_W - 6 - rot.shape[1], CARD_H - 6 - rot.shape[0]
    card[by:by + rot.shape[0], bx:bx + rot.shape[1]] = np.minimum(card[by:by + rot.shape[0], bx:bx + rot.shape[1]], rot)
    return card


def card_back():
    b = np.full((CARD_H, CARD_W, 3), 255, np.uint8)
    b[8:-8, 8:-8] = (140, 60, 20)
    for k in range(-CARD_H, CARD_W, 14):
        cv2.line(b, (8 + k, 8), (8 + k + CARD_H, 8 + CARD_H), (170, 90, 40), 3)
    b[:8], b[-8:], b[:, :8], b[:, -8:] = 255, 255, 255, 255
    return b


# ── animation ────────────────────────────────────────────────────────────────
class Sprite:
    def __init__(self, img, pos):
        self.img, self.pos, self.prev = img, np.array(pos, float), np.array(pos, float)
        self.path = []

    def move_to(self, dst, frames):
        start = self.path[-1] if self.path else self.pos
        for i in range(1, frames + 1):
            t = 1 - (1 - i / frames) ** 2                       # ease-out
            self.path.append(start + (np.array(dst, float) - start) * t)

    def step(self):
        self.prev = self.pos.copy()
        if self.path:
            self.pos = self.path.pop(0)


def composite(bg, sprites, alpha, offset=(0, 0)):
    out = bg.copy()
    H, W = out.shape[:2]
    for s in sprites:
        img, a = s.img, alpha.astype(np.float32) / 255
        v = s.pos - s.prev
        L = int(np.linalg.norm(v) * 0.5)
        if L >= 3:                                             # motion blur along the movement
            k = np.zeros((L | 1, L | 1), np.float32)
            c = (L | 1) // 2
            d = v / np.linalg.norm(v) * (L / 2)
            cv2.line(k, (int(c - d[0]), int(c - d[1])), (int(c + d[0]), int(c + d[1])), 1.0, 1)
            k /= k.sum()
            img = cv2.filter2D(img, -1, k)
            a = cv2.filter2D(a, -1, k)
        x, y = int(round(s.pos[0])), int(round(s.pos[1]))
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + CARD_W), min(H, y + CARD_H)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = img[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
        aa = a[y0 - y:y1 - y, x0 - x:x1 - x, None]
        out[y0:y1, x0:x1] = (sub * aa + out[y0:y1, x0:x1] * (1 - aa)).astype(np.uint8)
    if offset != (0, 0):                                       # window dragged
        M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
        out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
    return out


def build_video():
    bg, red_base, black_base, alpha = load_assets()
    card = lambda r, s: make_card(r, s, red_base, black_base)
    back = card_back()
    frames, truth = [], []
    P0, D0 = np.array([1046, 845]), np.array([1053, 287])
    D_STEP = np.array([45, 0])

    def run(sprites, n, offset=(0, 0), cover=None):
        for _ in range(n):
            for s in sprites:
                s.step()
            f = composite(bg, sprites, alpha, offset)
            if cover is not None:
                x, y, w, h = cover
                f[y:y + h, x:x + w] = (30, 30, 30)
            frames.append(f)

    def deal(sprites, rank_suit, dst, face_up=True, n=10):
        s = Sprite(card(*rank_suit) if face_up else back, SHOE)
        s.move_to(dst, n)
        sprites.append(s)
        run(sprites, n + 4)
        if face_up:
            truth.append(rank_suit)
        return s

    def sweep(sprites):
        for s in sprites:
            s.move_to(DISCARD, 12)
        run(sprites, 13)
        sprites.clear()
        run(sprites, 25)                                        # empty table between rounds

    run([], 15)
    # Round 1: player 8S 3S, dealer 8D + hole, player hits 5C, hole flips to KH
    sp = []
    deal(sp, ('8', 'Spades'), P0)
    deal(sp, ('8', 'Diamonds'), D0)
    deal(sp, ('3', 'Spades'), P0 + PILE_STEP)
    hole = deal(sp, None, D0 + D_STEP, face_up=False)
    run(sp, 15)
    deal(sp, ('5', 'Clubs'), P0 + 2 * PILE_STEP)
    run(sp, 6, cover=(1000, 650, 320, 420))                     # pop-up hides the pile for 6 frames
    run(sp, 10)
    hole.img = card('K', 'Hearts')
    truth.append(('K', 'Hearts'))
    run(sp, 25)
    sweep(sp)

    # Round 2: 8C 8H vs 6D, split (8H slides right), each hand gets a card, hole 10S, dealer draws 4H
    deal(sp, ('8', 'Clubs'), P0)
    deal(sp, ('6', 'Diamonds'), D0)
    p2 = deal(sp, ('8', 'Hearts'), P0 + PILE_STEP)
    deal(sp, None, D0 + D_STEP, face_up=False)
    hole = sp[-1]
    run(sp, 10)
    split_pos = P0 + np.array([260, 0])
    p2.move_to(split_pos, 12)
    run(sp, 16)
    deal(sp, ('2', 'Diamonds'), P0 + PILE_STEP)
    deal(sp, ('J', 'Clubs'), split_pos + PILE_STEP)
    hole.img = card('10', 'Spades')
    truth.append(('10', 'Spades'))
    run(sp, 10)
    deal(sp, ('4', 'Hearts'), D0 + 2 * D_STEP)
    run(sp, 20)
    sweep(sp)

    # Round 3: AS QD vs 5H + hole 7C; the window is dragged mid-round
    deal(sp, ('A', 'Spades'), P0)
    deal(sp, ('5', 'Hearts'), D0)
    deal(sp, ('Q', 'Diamonds'), P0 + PILE_STEP)
    hole = deal(sp, None, D0 + D_STEP, face_up=False)
    run(sp, 8)
    for i in range(1, 16):                                      # drag window by (-150, +60)
        run(sp, 1, offset=(int(-150 * i / 15), int(60 * i / 15)))
    run(sp, 10, offset=(-150, 60))
    hole.img = card('7', 'Clubs')
    truth.append(('7', 'Clubs'))
    run(sp, 20, offset=(-150, 60))
    frames.append(np.zeros_like(frames[-1]) + 20)
    return frames, truth


def main():
    frames, truth = build_video()
    reader = bj.TableReader(os.path.join(HERE, 'templates'))
    tracker = bj.CardTracker(num_decks=8, cfg=reader.cfg)
    size = (frames[0].shape[1] // 2, frames[0].shape[0] // 2)
    out = cv2.VideoWriter(os.path.join(HERE, 'tracking_test.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
    for f in frames:
        res = reader.process(f)
        tracker.update(res.detections)
        vis = bj.draw_frame(f, res, tracker)
        out.write(cv2.resize(vis, size, interpolation=cv2.INTER_AREA))
    out.release()

    expected_rc = sum(bj.HILO[r] for r, _ in truth)
    counted = sorted((r, s) for _, _, r, s, v in tracker.log if not isinstance(v, str))
    print(f"frames: {len(frames)}   cards dealt face up: {len(truth)}")
    print(f"expected running count {expected_rc:+d}   tracker: {tracker.running_count:+d}")
    print(f"cards counted {tracker.cards_seen} (expected {len(truth)})   rounds seen {tracker.round_no}")
    missing = sorted(set(truth) - set(counted))
    extra = [c for c in counted if counted.count(c) > truth.count(c)]
    if missing:
        print("  missed:", missing)
    if extra:
        print("  double-counted:", sorted(set(extra)))
    ok = tracker.running_count == expected_rc and sorted(truth) == counted
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

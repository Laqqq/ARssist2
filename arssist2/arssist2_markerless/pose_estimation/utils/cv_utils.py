from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "set_gamma",
    "refine_corners_subpix",
    "order_corners_clockwise",
    "order_corners_relative_to_line",
    "line_coefficients",
    "segment_intersection",
    "minimal_color_elimination",
]


def set_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply simple power-law gamma correction."""
    gamma = max(float(gamma), 0.01)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def refine_corners_subpix(
    gray: np.ndarray,
    corners: Sequence[Sequence[float]],
    win: Tuple[int, int] = (5, 5),
    zero_zone: Tuple[int, int] = (-1, -1),
    iters: int = 30,
) -> np.ndarray:
    """Refine corner coordinates in-place using cv2.cornerSubPix."""
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-4)
    corners_arr = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, corners_arr, win, zero_zone, term)
    return corners_arr.reshape(-1, 2)


def order_corners_clockwise(pts: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Return the four corners arranged in TL, TR, BR, BL order."""
    pts_arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    y_sorted = pts_arr[np.argsort(pts_arr[:, 1])]
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]
    tl, tr = top_two[np.argsort(top_two[:, 0])]
    bl, br = bottom_two[np.argsort(bottom_two[:, 0])]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def order_corners_relative_to_line(
    pts: Sequence[Sequence[float]],
    line_abc: Tuple[float, float, float],
) -> list[tuple[float, float]]:
    """Order square corners relative to a separator line."""
    pts_arr = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    A, B, C = line_abc
    distances = A * pts_arr[:, 0] + B * pts_arr[:, 1] + C
    idx = np.argsort(np.abs(distances))
    bottom_idx = idx[:2]
    top_idx = idx[2:]
    bottom_pts = pts_arr[bottom_idx]
    top_pts = pts_arr[top_idx]

    horizontal = abs(B) > abs(A)
    if horizontal:
        top_pts = top_pts[np.argsort(top_pts[:, 0])]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
    else:
        centroid_x = pts_arr[:, 0].mean()
        x_line = -C / A if A != 0 else centroid_x
        square_right = centroid_x > x_line
        if square_right:
            sort_top = np.argsort(top_pts[:, 1])
            sort_bottom = np.argsort(bottom_pts[:, 1])
        else:
            sort_top = np.argsort(-top_pts[:, 1])
            sort_bottom = np.argsort(-bottom_pts[:, 1])
        top_pts = top_pts[sort_top]
        bottom_pts = bottom_pts[sort_bottom]

    tl, tr = top_pts
    bl, br = bottom_pts
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def line_coefficients(p1: Sequence[float], p2: Sequence[float]) -> Tuple[float, float, float]:
    """Return coefficients (A, B, C) of the line passing through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    return A, B, C


def segment_intersection(
    segment: Tuple[Sequence[float], Sequence[float]],
    infinite_line: Tuple[Sequence[float], Sequence[float]],
) -> Tuple[float, float] | None:
    """Return the intersection between a segment and a line if it lies on the segment."""
    A1, B1, C1 = line_coefficients(*segment)
    A2, B2, C2 = line_coefficients(*infinite_line)
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det

    (sx1, sy1), (sx2, sy2) = segment
    minx, maxx = sorted((sx1, sx2))
    miny, maxy = sorted((sy1, sy2))
    if minx - 1e-3 <= x <= maxx + 1e-3 and miny - 1e-3 <= y <= maxy + 1e-3:
        return (float(x), float(y))
    return None


def minimal_color_elimination(img_hsv: np.ndarray) -> np.ndarray:
    """Return a binary mask isolating the dominant hue band."""
    h_channel, s_channel, v_channel = cv2.split(img_hsv)
    hist = cv2.calcHist([h_channel], [0], None, [180], (0, 180))
    hist = cv2.normalize(hist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    peak_hue = int(np.argmax(hist))
    interval = 10
    lower_hue = peak_hue - interval
    upper_hue = peak_hue + interval

    def hue_mask(lower: int, upper: int) -> np.ndarray:
        return cv2.inRange(h_channel, lower, upper)

    if lower_hue < 0:
        mask = cv2.bitwise_or(hue_mask(0, upper_hue), hue_mask(lower_hue + 180, 179))
    elif upper_hue >= 180:
        mask = cv2.bitwise_or(hue_mask(lower_hue, 179), hue_mask(0, upper_hue - 180))
    else:
        mask = hue_mask(lower_hue, upper_hue)

    sat_mask = cv2.inRange(s_channel, 140, 255)
    val_mask = cv2.inRange(v_channel, 140, 255)
    return cv2.bitwise_and(cv2.bitwise_and(mask, sat_mask), val_mask)


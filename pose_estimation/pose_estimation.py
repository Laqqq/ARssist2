"""Pose-estimation pipeline for the Cornerstone instrument."""

import itertools
import math

import cv2
import numpy as np
from ultralytics import YOLO

from utils.cv_utils import *
from utils.drawing_utils import draw_pose_coord_frame_in_image

ROI_TARGET_AREA = 10000.0
HULL_OFFSET = 40
MIN_CONTOUR_AREA_RATIO = 0.01
MAX_CONTOUR_AREA_RATIO = 0.45
EDGE_K = 10
EDGE_THR = -0.95
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)
SQUARE_WIDTH = 0.01926
SQUARE_HEIGHT = 0.01926
LINE_NEAR_Y = -0.02224
LINE_FAR_Y = -0.0495


class PoseEstimationError(RuntimeError):
    """Raised when any step of the pipeline cannot be completed."""


def process_image(img_rgb, camera_matrix, instrument_type, model):
    """
    High-level entrypoint of the pose estimation pipeline.
    """
    try:
        return process_image_implementation(img_rgb, camera_matrix, instrument_type, model)
    except PoseEstimationError as exc:
        print(f"[pose_estimation] {exc}")
        return (None, None, None, None, None, None, None, None)



def detect_primary_box(model, img_rgb):
    "Detect the primary box of the instrument based on the YOLO and the image."
    results = model(img_rgb)
    if not results or not hasattr(results[0], "boxes") or len(results[0].boxes) == 0:
        return None, None, None, None
    boxes = results[0].boxes
    confs = boxes.conf
    best_idx = int(np.argmax(confs))
    best_box = boxes[best_idx]
    x_min, y_min, x_max, y_max = map(int, best_box.xyxy[0])
    return x_min, y_min, x_max, y_max


def extract_zoom(img_rgb, bbox, shrink_ratio=0.95):
    "Extract the zoomed image based on the bounding box."
    x_min, y_min, x_max, y_max = bbox
    crop_bottom = int(y_max + 0.7 * abs(y_max - y_min))
    crop_bottom = min(crop_bottom, img_rgb.shape[0])
    cropped = img_rgb[y_min:crop_bottom, x_min:x_max]
    if cropped.size == 0:
        return None, None, None
    h, w = cropped.shape[:2]
    new_h = max(int(h * shrink_ratio), 1)
    new_w = max(int(w * shrink_ratio), 1)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    bottom = top + new_h
    right = left + new_w
    zoomed = cropped[top:bottom, left:right]
    if zoomed.size == 0:
        return None, None, None
    offset_x = x_min + left
    offset_y = y_min + top
    return zoomed, offset_x, offset_y


def build_button_mask(zoomed_img, instrument_type):
    "Build the button mask based on the zoomed image and the instrument type."
    if instrument_type == "xi":
        hsv = cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2HSV)
        mask = minimal_color_elimination(hsv)
        return mask, mask
    gray = cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0.0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)
    return mask, blurred


def locate_square_contour(mask, roi_area, image_shape):
    "Locate the square contour based on the mask, the ROI area and the image shape."
    h, w = image_shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    min_area = MIN_CONTOUR_AREA_RATIO * roi_area
    max_area = MAX_CONTOUR_AREA_RATIO * roi_area
    best_cnt, best_score = None, -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        (_, _), (wc, hc), _ = cv2.minAreaRect(c)
        if wc >= 0.5 * w or hc >= 0.5 * h:
            continue
        pts = c.reshape(-1, 2)
        if (
            (pts[:, 0] == 0).any()
            or (pts[:, 0] == w - 1).any()
            or (pts[:, 1] == 0).any()
            or (pts[:, 1] == h - 1).any()
        ):
            continue
        score = rectangle_score(c)
        if score > best_score:
            best_cnt, best_score = c, score
    return best_cnt


def rectangle_score(cnt):  
    "Compute the score of the rectangle based on the contour."
    area = cv2.contourArea(cnt)
    if area <= 0:
        return 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    if w == 0 or h == 0:
        return 0.0
    ar = min(w, h) / max(w, h) # the aspect ratio of the rectangle
    extent = area / (w * h) # the extent of the rectangle
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / (hull_area + 1e-6) # the solidity of the rectangle
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True).reshape(-1, 2)
    if not (4 <= len(approx) <= 8):
        return 0.0

    def angle(p0, p1, p2):
        v1, v2 = p0 - p1, p2 - p1
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cosang = np.dot(v1, v2) / denom
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    ang_err = np.mean(
        [
            abs(angle(approx[(i - 1) % len(approx)], approx[i], approx[(i + 1) % len(approx)]) - 90)
            for i in range(len(approx)) # the angle between the adjacent edges
        ]
    )
    ortho = 1 - ang_err / 90
    return max(0.0, (ar * extent * solidity * ortho) ** 0.25) # the score of the rectangle


def scale_hull(hull, rect, target_area=ROI_TARGET_AREA, offset=HULL_OFFSET):
    "Scale the hull of the square based on the rectangle and the target area."
    x0, y0, w0, h0 = rect
    area = max(cv2.contourArea(hull), 1e-6)
    scale = math.sqrt(target_area / area)
    hull_f = hull.astype(np.float32)
    scaled = (
        hull_f * scale
        - scale * np.array([x0, y0], dtype=np.float32)
        + np.array([offset, offset], dtype=np.float32)
    )
    return scaled.reshape(-1, 1, 2), scale, offset, (x0, y0)


def densify_contour(hull_scaled):
    "Densify the contour of the square."
    pts = hull_scaled.reshape(-1, 2)
    dense_pts = []
    for p_curr, p_next in zip(pts, np.roll(pts, -1, axis=0)):
        dist = np.linalg.norm(p_next - p_curr)
        n_samples = max(int(dist), 1)
        for t in np.linspace(0, 1, n_samples, endpoint=False):
            dense_pts.append(p_curr * (1 - t) + p_next * t)
    return np.asarray(dense_pts, dtype=np.float32)


def extract_edge_points(hull_dense, k=EDGE_K, thr=EDGE_THR):
    "Remove the corner points and extract the edge points of the square."
    edge_pts = []
    N = len(hull_dense)
    for i, p in enumerate(hull_dense):
        p0 = hull_dense[(i - k) % N]
        p2 = hull_dense[(i + k) % N]
        v1 = p0 - p
        v2 = p2 - p
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cosang = np.dot(v1, v2) / denom
        if cosang < thr:
            edge_pts.append(p)
    return np.asarray(edge_pts, dtype=np.float32)


def ransac_edges_with_lines(edge_pts, num_edges=4, n_iter=100, eps=2.0, min_inliers=6):  
    "RANSAC to find the edges of the square using line fitting."
    pts = edge_pts.astype(np.float32)
    N = pts.shape[0]
    remaining = np.arange(N)
    labels = -np.ones(N, np.int32)
    lines = []
    # RANSAC to find the edges of the square using line fitting
    for edge_id in range(num_edges):
        if remaining.size < min_inliers:
            break
        pair_idx = np.random.choice(remaining, size=(n_iter, 2), replace=True)
        p1 = pts[pair_idx[:, 0]]
        p2 = pts[pair_idx[:, 1]]
        good = np.any(p1 != p2, axis=1)
        p1, p2 = p1[good], p2[good]
        if p1.size == 0:
            break
        dx, dy = (p2 - p1).T
        norm = np.hypot(dx, dy)
        nx, ny = dy / norm, -dx / norm
        C = -(nx * p1[:, 0] + ny * p1[:, 1])

        pr = pts[remaining]
        d = np.abs(pr[:, 0, None] * nx + pr[:, 1, None] * ny + C)
        inlier_mask = d < eps
        inlier_cnt = inlier_mask.sum(0)

        best_idx = np.argmax(inlier_cnt)
        best_cnt = inlier_cnt[best_idx]
        if best_cnt < min_inliers:
            break

        best_inliers = remaining[inlier_mask[:, best_idx]]
        pts_in = pts[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(
            pts_in, cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        lines.append((vx, vy, x0, y0))
        labels[best_inliers] = edge_id
        remaining = np.setdiff1d(remaining, best_inliers, assume_unique=True)

    if len(lines) != num_edges:
        return None, None, None

    corners = []
    angle_thresh = 10.0
    for l1, l2 in itertools.combinations(lines, 2):
        v1 = np.array(l1[:2])
        v2 = np.array(l2[:2])
        cos_angle = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
        if angle_deg < angle_thresh:
            continue
        corner = intersect_line_pair(l1, l2)
        if corner is not None:
            corners.append(corner)

    if len(corners) < 4:
        return None, None, None
    return labels, lines, corners


def intersect_line_pair(line1, line2):
    "Intersect two lines."
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2
    a1, b1, c1 = vy1, -vx1, vy1 * x1 - vx1 * y1
    a2, b2, c2 = vy2, -vx2, vy2 * x2 - vx2 * y2
    det = a1 * b2 - a2 * b1
    if abs(det) < 0.1:
        return None
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    return (x, y)


def render_mask_from_points(hull_scaled, rect, scale, offset):
    "Render the mask from the points."
    _, _, w0, h0 = rect
    mask_height = int(scale * h0) + 2 * offset
    mask_width = int(scale * w0) + 2 * offset
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    hull_int = np.round(hull_scaled).astype(np.int32)
    cv2.drawContours(mask, [hull_int], -1, 255, thickness=1)
    return mask


def restore_corners(points, scale, offset, origin):
    "Restore the corners of the square based on the normalization metadata."
    restored = []
    x0, y0 = origin
    for pt in points:
        xs, ys = pt
        restored.append((x0 + (xs - offset) / scale, y0 + (ys - offset) / scale))
    return restored


def find_separator_lines(gray_img, hull, corners):
    "Find the separator lines of the instrument based on the hull and the corners."
    MIN_VERT_ANG = 30.0
    pts = hull.reshape(-1, 2).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    cx, cy = x + w / 2.0, y + h / 2.0

    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    edges_clean = edges.copy()
    edges_clean[y : y + h, x : x + w] = 0
    lines = cv2.HoughLines(edges_clean, 1, np.pi / 180, int(1.3 * w))
    if lines is None:
        raise PoseEstimationError("Hough detector found no separator lines.")

    edge_pts = np.column_stack(np.where(edges_clean > 0))
    candidates = []
    H, W = gray_img.shape[:2]
    diag = math.hypot(W, H)
    # find the separator lines
    for rho_theta in lines[:, 0]:
        rho, theta = rho_theta
        angle_deg = math.degrees(theta)
        if angle_deg < MIN_VERT_ANG or angle_deg > 180.0 - MIN_VERT_ANG:
            continue
        A = math.cos(theta)
        B = math.sin(theta)
        C = -rho
        mask = np.abs(A * edge_pts[:, 1] + B * edge_pts[:, 0] + C) <= 1.0 # the mask of the inliers
        votes = int(mask.sum()) # the number of inliers
        if votes == 0:
            continue
        dist_center = abs(A * cx + B * cy + C)
        candidates.append(
            {
                "rho": float(rho),
                "theta": float(theta),
                "votes": votes,
                "dist_center": dist_center,
                "A": A,
                "B": B,
                "C": C,
            }
        )

    if not candidates:
        raise PoseEstimationError("No valid separator line candidates.")
    candidates.sort(key=lambda c: (-c["votes"], c["dist_center"]))
    best = candidates[0]

    filtered = [
        c for c in candidates[1:] if c["dist_center"] > 1.45 * best["dist_center"]
    ]
    A, B, C = best["A"], best["B"], best["C"]
    D = A * cx + B * cy + C
    fx = cx - A * D
    fy = cy - B * D
    vx, vy = fx - cx, fy - cy
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        raise PoseEstimationError("Cannot derive direction for secondary line search.")
    ux, uy = vx / norm, vy / norm
    intersecting = []
    for cand in filtered:
        denom = cand["A"] * ux + cand["B"] * uy
        if abs(denom) < 0.9:
            continue
        u = -(cand["A"] * cx + cand["B"] * cy + cand["C"]) / denom
        if u <= 0:
            continue
        intersecting.append(cand)
    if not intersecting:
        raise PoseEstimationError("Failed to find a secondary separator line.")
    second = max(intersecting, key=lambda c: c["votes"])

    return [
        rho_theta_to_segment(best, diag),
        rho_theta_to_segment(second, diag),
    ]


def rho_theta_to_segment(candidate, diag):
    "Convert the rho and theta to a segment."
    r = candidate["rho"]
    t = candidate["theta"]
    x0 = math.cos(t) * r
    y0 = math.sin(t) * r
    dx = -math.sin(t)
    dy = math.cos(t)
    pt1 = (x0 + dx * diag, y0 + dy * diag)
    pt2 = (x0 - dx * diag, y0 - dy * diag)
    return (pt1[0], pt1[1], pt2[0], pt2[1])


def normalized_line_params(line):
    x1, y1, x2, y2 = line
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    norm = math.hypot(A, B)
    if norm == 0:
        raise PoseEstimationError("Invalid line for normalization.")
    return A / norm, B / norm, C / norm


def compute_line_intersections(ordered_corners, lines):
    "Compute the intersections of the ordered corners with the separator lines."
    segments = [
        (ordered_corners[i], ordered_corners[(i + 1) % 4]) for i in range(4)
    ]
    intersections = []
    for line in lines:
        p_line = ((line[0], line[1]), (line[2], line[3]))
        pts = []
        for seg in segments:
            pt = segment_intersection(seg, p_line)
            if pt:
                pts.append(pt)
        if len(pts) != 2:
            raise PoseEstimationError("Failed to intersect separator line with square.")
        pts = sorted(pts, key=lambda p: p[0])
        intersections.extend(pts)
    return intersections


def local_to_global(points, offset_x, offset_y):
    "Convert the local points to global points."
    return [(x + offset_x, y + offset_y) for x, y in points]


def object_points():
    "Return the object points for the square."
    width = SQUARE_WIDTH
    height = SQUARE_HEIGHT
    return np.array(
        [
            [-width / 2, height / 2, 0],
            [width / 2, height / 2, 0],
            [width / 2, -height / 2, 0],
            [-width / 2, -height / 2, 0],
            [-width / 2, LINE_NEAR_Y, 0],
            [width / 2, LINE_NEAR_Y, 0],
            [-width / 2, LINE_FAR_Y, 0],
            [width / 2, LINE_FAR_Y, 0],
        ],
        dtype=np.float32,
    )


def augment_correspondences(obj_pts, img_pts, samples_per_edge=3):
    "Augment the correspondences by adding more points along the edges of the square."
    def linspace_rows(p0, p1, count):
        if count <= 0:
            return np.empty((0, p0.shape[-1]), dtype=np.float32)
        return np.linspace(p0, p1, count + 2, dtype=np.float32)[1:-1]
    # add more points along the edges of the square
    pairs = [
        ((0, 1), samples_per_edge),
        ((1, 2), samples_per_edge),
        ((2, 3), samples_per_edge),
        ((3, 0), samples_per_edge),
        ((4, 5), samples_per_edge),
        ((6, 7), samples_per_edge),
        ((4, 6), samples_per_edge * 2),
        ((5, 7), samples_per_edge * 2),
    ]
    extra_obj = []
    extra_img = []
    for (i, j), count in pairs:
        obj_line = linspace_rows(obj_pts[i], obj_pts[j], count)
        img_line = linspace_rows(img_pts[i], img_pts[j], count)
        if obj_line.size and img_line.size:
            extra_obj.append(obj_line)
            extra_img.append(img_line)

    if extra_obj:
        obj_aug = np.vstack([obj_pts] + extra_obj)
        img_aug = np.vstack([img_pts] + extra_img)
    else:
        obj_aug = obj_pts
        img_aug = img_pts
    return obj_aug.astype(np.float32), img_aug.astype(np.float32)


def solve_pnp_with_selection(obj_pts, img_pts, obj_pts_full, img_pts_full, camera_matrix, view_direction):
    "Solve the PnP problem using the generic solver and select the pose candidate based on the view direction."
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj_pts, img_pts, camera_matrix, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok or len(rvecs) == 0:
        raise PoseEstimationError("cv2.solvePnPGeneric did not return a pose.")

    refined = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        if np.linalg.det(R) < 0.0:
            continue
        rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
            obj_pts_full, img_pts_full, camera_matrix, DIST_COEFFS, rvec, tvec
        )
        refined.append((rvec_ref, tvec_ref))
    if not refined:
        raise PoseEstimationError("Pose refinement rejected all candidates.")
    return select_pose_candidate(refined, obj_pts, img_pts, camera_matrix, view_direction)


def select_pose_candidate(candidates, obj_pts, img_pts, camera_matrix, view_direction, steep_thresh=55.0):
    "Select the pose candidate based on the view direction,the steepness of the line and the camera matrix."
    axis = np.array([[0, 0, 0.0], [0, 0, 0.05]], dtype=np.float32)
    fallback_rvecs = []
    fallback_tvecs = []
    for rvec, tvec in candidates:
        fallback_rvecs.append(rvec)
        fallback_tvecs.append(tvec)
        pts2d, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, DIST_COEFFS) # project the axis onto the image
        (ox, oy), (zx, zy) = pts2d.reshape(2, 2)    
        dx = float(zx - ox) # the difference in the x-axis
        dy = float(zy - oy) # the difference in the y-axis
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))) # the angle between the axis and the line
        # if the angle is greater than the steep threshold, use the EPNP solver
        if angle > steep_thresh:
            success, rvec_epnp, tvec_epnp = cv2.solvePnP(
                obj_pts, img_pts, camera_matrix, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP
            )
            if success:
                return cv2.solvePnPRefineLM(
                    obj_pts, img_pts, camera_matrix, DIST_COEFFS, rvec_epnp, tvec_epnp
                )
        if view_direction == "view_left" and dx < 0:
            return rvec, tvec
        if view_direction == "view_right" and dx > 0:
            return rvec, tvec
    return choose_pose_facing_camera(fallback_rvecs, fallback_tvecs)


def choose_pose_facing_camera(rvecs, tvecs):
    "Choose the pose facing the camera based on the score of the rotation matrix."
    sel_idx = None
    best_score = np.inf
    for idx, (rvec, _) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rvec)
        score = R[2, 2] # the z-axis of the rotation matrix
        if score < best_score:
            best_score = score
            sel_idx = idx
    if sel_idx is None:
        raise PoseEstimationError("Unable to choose pose facing the camera.")
    return rvecs[sel_idx], tvecs[sel_idx]


def draw_results(base_img, img_pts, rvec, tvec, camera_matrix):
    out_img = base_img.copy()
    for i in range(4):
        p1 = tuple(img_pts[i].astype(int))
        p2 = tuple(img_pts[(i + 1) % 4].astype(int))
        cv2.line(out_img, p1, p2, (255, 0, 0), 1)

    shaft = np.array(
        [
            [-0.03303, -0.03094, -0.0275],
            [-0.03303, -0.570, -0.0275],
        ],
        dtype=np.float32,
    )
    shaft_sel = np.array(
        [
            [-0.03303, -0.03094, -0.0275],
            [-0.03303, -0.170, -0.0275],
        ],
        dtype=np.float32,
    )
    shaft_img, _ = cv2.projectPoints(shaft, rvec, tvec, camera_matrix, DIST_COEFFS)
    shaft_sel_img, _ = cv2.projectPoints(shaft_sel, rvec, tvec, camera_matrix, DIST_COEFFS)
    s1, s2 = map(tuple, shaft_sel_img.reshape(-1, 2).astype(int))
    cv2.line(out_img, s1, s2, (0, 255, 0), 3)
    center_pt = s2
    cv2.circle(out_img, center_pt, 5, (0, 0, 255), -1)

    R_sel, _ = cv2.Rodrigues(rvec)
    out_img = draw_pose_coord_frame_in_image(R_sel, tvec, camera_matrix, out_img, scale=0.08, brightness=255)

    shaft_line, _ = cv2.projectPoints(shaft, rvec, tvec, camera_matrix, DIST_COEFFS)
    shaft_line_pts = shaft_line.reshape(-1, 2).astype(int)
    cv2.line(out_img, tuple(shaft_line_pts[0]), tuple(shaft_line_pts[1]), (255, 255, 0), 1)

    left_line, right_line, edge_left_3d_world, edge_right_3d_world = cylinder_outline_lines(
        shaft[0], shaft[1], 0.0043, R_sel, tvec.reshape(3,), camera_matrix
    )
    l1, l2 = [tuple(pt.astype(int)) for pt in left_line]
    r1, r2 = [tuple(pt.astype(int)) for pt in right_line]
    cv2.line(out_img, l1, l2, (0, 255, 255), 2)
    cv2.line(out_img, r1, r2, (0, 255, 255), 2)

    shaft3d = shaft.copy()
    point3d = np.column_stack(
        (
            np.hstack((shaft3d[0], 1.0)),
            np.hstack((shaft3d[1], 1.0)),
            np.hstack((edge_left_3d_world[0], 1.0)),
            np.hstack((edge_left_3d_world[1], 1.0)),
            np.hstack((edge_right_3d_world[0], 1.0)),
            np.hstack((edge_right_3d_world[1], 1.0)),
        )
    )
    return out_img, center_pt, point3d


def cylinder_outline_lines(P1, P2, R_cyl, R_wc, t_wc, K):
    "Compute the two silhouette lines (left & right) of a long, thin cylinder."
    def world_to_cam(P):
        return R_wc @ P + t_wc

    def cam_to_pix(Xc):
        u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
        v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
        return np.array([u, v], dtype=np.float64)

    C1, C2 = world_to_cam(P1), world_to_cam(P2)
    d_cam = C2 - C1
    d_cam /= np.linalg.norm(d_cam)
    view_vec = 0.5 * (C1 + C2)
    n_cam = np.cross(d_cam, view_vec)
    if np.linalg.norm(n_cam) < 1e-8:
        n_cam = np.cross(d_cam, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(n_cam) < 1e-8:
            n_cam = np.cross(d_cam, np.array([0.0, 1.0, 0.0]))
    n_cam /= np.linalg.norm(n_cam)
    # compute the left and right edges of the cylinder
    L1_cam = C1 + R_cyl * n_cam
    L2_cam = C2 + R_cyl * n_cam
    R1_cam = C1 - R_cyl * n_cam
    R2_cam = C2 - R_cyl * n_cam

    edge_left_3d_cam = np.vstack((L1_cam, L2_cam))
    edge_right_3d_cam = np.vstack((R1_cam, R2_cam))
    edge_left_3d_world = (R_wc.T @ (edge_left_3d_cam.T - t_wc[:, None])).T
    edge_right_3d_world = (R_wc.T @ (edge_right_3d_cam.T - t_wc[:, None])).T

    l_left = (cam_to_pix(L1_cam), cam_to_pix(L2_cam))
    l_right = (cam_to_pix(R1_cam), cam_to_pix(R2_cam))
    return l_left, l_right, edge_left_3d_world, edge_right_3d_world


def decide_viewpoint(line, hull, image_width, slope_thresh_deg=60.0):
    "Determine the viewpoint of the instrument based on the line and the rectangle button."    
    (x1, y1, x2, y2) = line
    hull_pts = hull.reshape(-1, 2)
    centroid = hull_pts.mean(axis=0)
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    angle = min(angle, 180.0 - angle)
    if angle <= slope_thresh_deg:
        return "view_left" if centroid[0] <= 0.5 * image_width else "view_right"
    mid_x = 0.5 * (x1 + x2)
    return "view_left" if mid_x < centroid[0] else "view_right"


def process_image_implementation(img_rgb, camera_matrix, instrument_type, model):
    if img_rgb is None:
        raise PoseEstimationError("Input image is None.")
    if instrument_type not in {"xi", "cornerstone"}:
        raise PoseEstimationError(f"Unsupported instrument type: {instrument_type}")

    base_img = img_rgb.copy()
    bbox = detect_primary_box(model, base_img)
    if bbox is None:
        raise PoseEstimationError("Object detector returned no bounding boxes.")
    # gamma correction and zooming
    work_img = set_gamma(base_img, gamma=0.95)
    zoomed_data = extract_zoom(work_img, bbox)
    if zoomed_data is None:
        raise PoseEstimationError("Failed to crop zoomed ROI.")
    zoomed_img, offset_x, offset_y = zoomed_data
    roi_area = float(zoomed_img.shape[0] * zoomed_img.shape[1])
    # build the button mask
    mask, blurred = build_button_mask(zoomed_img, instrument_type)
    contour = locate_square_contour(mask, roi_area, zoomed_img.shape[:2])
    if contour is None:
        raise PoseEstimationError("No valid square contour detected.")
    # convex hull
    hull = cv2.convexHull(contour, returnPoints=True)
    rect = cv2.boundingRect(hull)
    hull_scaled, hull_scale, hull_offset, hull_origin = scale_hull(hull, rect)
    # densify the contour
    hull_dense = densify_contour(hull_scaled)
    edge_pts = extract_edge_points(hull_dense)
    # ransac to find the edges of the square
    labels, lines, scaled_corners = ransac_edges_with_lines(edge_pts)
    if labels is None or len(scaled_corners) != 4:
        raise PoseEstimationError("RANSAC failed to recover four square edges.")
    # refine the corners
    scaled_corners = refine_corners_subpix(
        render_mask_from_points(hull_scaled, rect, hull_scale, hull_offset),
        np.asarray(scaled_corners, dtype=np.float32),
    )
    orig_corners = restore_corners(scaled_corners, hull_scale, hull_offset, hull_origin)
    separator_lines = find_separator_lines(blurred, hull, orig_corners)
    if len(separator_lines) < 2:
        raise PoseEstimationError("Unable to find both separator lines.")
    # compute the intersections of the ordered corners with the separator lines
    first_line = separator_lines[0]
    A, B, C = normalized_line_params(first_line)
    ordered_corners = order_corners_relative_to_line(orig_corners, (A, B, C))
    intersections = compute_line_intersections(ordered_corners, separator_lines)
    final_pts_local = ordered_corners + intersections
    final_pts_global = local_to_global(final_pts_local, offset_x, offset_y)
    # augment the correspondences
    obj_pts = object_points()
    img_pts = np.asarray(final_pts_global, dtype=np.float32)
    obj_pts_full, img_pts_full = augment_correspondences(obj_pts, img_pts)
    # decide the viewpoint  
    view_direction = decide_viewpoint(first_line, hull, zoomed_img.shape[1])
    # solve the PnP problem with the selection
    rvec, tvec = solve_pnp_with_selection(
        obj_pts, img_pts, obj_pts_full, img_pts_full, camera_matrix, view_direction
    )
    # draw the results
    out_img, center_pt, point3d = draw_results(
        base_img,
        img_pts,
        rvec,
        tvec,
        camera_matrix,
    )

    return out_img, center_pt, point3d, rvec, tvec, camera_matrix, obj_pts, img_pts



import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, jsonify, render_template_string

# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_NUMBER = 0

# Original calibration preserved
MM_PER_PIXEL = 0.36339648

# Camera inspection area in the original camera frame
X1, Y1 = 150, 70
X2, Y2 = 650, 570

# Red part detection
MIN_PART_AREA = 1000

# Inner square detection
MIN_HOLE_AREA_RATIO = 0.10
MIN_SQUARE_RATIO = 0.60
MAX_CENTER_OFFSET_RATIO = 0.30
MIN_INNER_DIMENSION = 20

# Cylinder detection
WHITE_MIN_BRIGHTNESS = 100
WHITE_MAX_SATURATION = 150
MIN_CYLINDER_AREA_RATIO = 0.08
MIN_CIRCULARITY = 0.40
MIN_CYLINDER_RADIUS = 5
CYLINDER_INSET_RATIO = 0.06
CYLINDER_MIN_AREA = 30

# Fit-gap threshold preserved from your current code
MAX_FIT_GAP_MM2 = 160

# Display colors (BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLUE = (255, 120, 0)
PURPLE = (255, 0, 255)
ORANGE = (0, 165, 255)

# ============================================================
# SHARED DATA
# ============================================================

app = Flask(__name__)
data_lock = threading.Lock()
latest_jpeg = None

measurements = {
    "camera": "STARTING",
    "part_detected": False,
    "inner_square_detected": False,
    "length": None,
    "width": None,
    "inner_area": None,
    "top_wall": None,
    "bottom_wall": None,
    "left_wall": None,
    "right_wall": None,
    "cylinder_detected": False,
    "cylinder_diameter": None,
    "cylinder_area": None,
    "fit_gap": None,
    "fit_state": "WAITING",
    "updated": time.time()
}


def update_measurements(**kwargs):
    with data_lock:
        measurements.update(kwargs)
        measurements["updated"] = time.time()


def empty_measurements():
    return {
        "camera": "LIVE",
        "part_detected": False,
        "inner_square_detected": False,
        "length": None,
        "width": None,
        "inner_area": None,
        "top_wall": None,
        "bottom_wall": None,
        "left_wall": None,
        "right_wall": None,
        "cylinder_detected": False,
        "cylinder_diameter": None,
        "cylinder_area": None,
        "fit_gap": None,
        "fit_state": "WAITING"
    }


# ============================================================
# ROTATION-AWARE GEOMETRY
# ============================================================

def get_rotated_box(contour):
    """
    Returns the rotated rectangle corners and its two local axes.
    The axes follow the object's orientation in the image.
    """
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(np.float32)

    edge_a = corners[1] - corners[0]
    edge_b = corners[3] - corners[0]

    len_a = np.linalg.norm(edge_a)
    len_b = np.linalg.norm(edge_b)

    if len_a <= 0 or len_b <= 0:
        return None

    axis_a = edge_a / len_a
    axis_b = edge_b / len_b

    return rect, corners, axis_a, axis_b


def project_contour(contour, axis_a, axis_b):
    """Projects contour points onto the block's local axes."""
    points = contour.reshape(-1, 2).astype(np.float32)

    projection_a = points @ axis_a
    projection_b = points @ axis_b

    return projection_a, projection_b


def point_from_local(a, b, axis_a, axis_b):
    """Converts coordinates in the local axes back to image pixels."""
    point = axis_a * a + axis_b * b
    return tuple(np.round(point).astype(int))


def draw_rotated_outline(display, contour, color, thickness=2):
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(np.int32)

    cv2.polylines(
        display,
        [corners],
        True,
        color,
        thickness,
        cv2.LINE_AA
    )

    return corners


# ============================================================
# ROTATION-AWARE WALL MEASUREMENTS
# ============================================================

def measure_four_walls(outer_contour, inner_contour):
    """
    Measures four wall thicknesses along the rotated block axes.

    This is more rotation-aware than measuring boundingRect offsets.
    It assumes the outer block and inner opening have approximately
    parallel corresponding edges.
    """
    geometry = get_rotated_box(outer_contour)

    if geometry is None:
        return None

    _, _, axis_a, axis_b = geometry

    outer_a, outer_b = project_contour(
        outer_contour, axis_a, axis_b
    )

    inner_a, inner_b = project_contour(
        inner_contour, axis_a, axis_b
    )

    oa_min, oa_max = outer_a.min(), outer_a.max()
    ob_min, ob_max = outer_b.min(), outer_b.max()

    ia_min, ia_max = inner_a.min(), inner_a.max()
    ib_min, ib_max = inner_b.min(), inner_b.max()

    # Confirm that the inner opening is within the outer contour.
    if (
        ia_min <= oa_min or ia_max >= oa_max or
        ib_min <= ob_min or ib_max >= ob_max
    ):
        return None

    walls_px = {
        "Left": ia_min - oa_min,
        "Right": oa_max - ia_max,
        "Top": ib_min - ob_min,
        "Bottom": ob_max - ib_max
    }

    if any(value <= 0 for value in walls_px.values()):
        return None

    return {
        name: {
            "thickness_px": float(value),
            "thickness_mm": float(value * MM_PER_PIXEL)
        }
        for name, value in walls_px.items()
    }


def draw_wall_measurements(display, outer_contour, inner_contour):
    result = measure_four_walls(
        outer_contour, inner_contour
    )

    if result is None:
        return None

    geometry = get_rotated_box(outer_contour)

    if geometry is None:
        return None

    _, outer_corners, axis_a, axis_b = geometry

    inner_geometry = get_rotated_box(inner_contour)

    if inner_geometry is None:
        return None

    _, inner_corners, _, _ = inner_geometry

    cv2.polylines(
        display, [outer_corners.astype(np.int32)],
        True, BLUE, 2, cv2.LINE_AA
    )

    cv2.polylines(
        display, [inner_corners.astype(np.int32)],
        True, GREEN, 2, cv2.LINE_AA
    )

    outer_a, outer_b = project_contour(
        outer_contour, axis_a, axis_b
    )

    inner_a, inner_b = project_contour(
        inner_contour, axis_a, axis_b
    )

    oa_min, oa_max = outer_a.min(), outer_a.max()
    ob_min, ob_max = outer_b.min(), outer_b.max()

    ia_min, ia_max = inner_a.min(), inner_a.max()
    ib_min, ib_max = inner_b.min(), inner_b.max()

    mid_a = (oa_min + oa_max) / 2
    mid_b = (ob_min + ob_max) / 2

    # Each measurement line crosses its corresponding wall.
    wall_lines = [
        (
            "Left",
            (oa_min, mid_b),
            (ia_min, mid_b)
        ),
        (
            "Right",
            (ia_max, mid_b),
            (oa_max, mid_b)
        ),
        (
            "Top",
            (mid_a, ob_min),
            (mid_a, ib_min)
        ),
        (
            "Bottom",
            (mid_a, ib_max),
            (mid_a, ob_max)
        )
    ]

    h, w = display.shape[:2]

    for label, start, end in wall_lines:
        p1 = point_from_local(
            start[0], start[1], axis_a, axis_b
        )

        p2 = point_from_local(
            end[0], end[1], axis_a, axis_b
        )

        cv2.line(
            display, p1, p2, YELLOW, 2, cv2.LINE_AA
        )

        value = result[label]["thickness_mm"]

        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2

        tx = max(3, min(w - 125, mx + 5))
        ty = max(16, min(h - 5, my - 5))

        cv2.putText(
            display,
            f"{label}: {value:.2f} mm",
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            YELLOW,
            1,
            cv2.LINE_AA
        )

    return result


# ============================================================
# RED PART DETECTION
# ============================================================

def detect_red_part(hsv):
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([165, 70, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    return mask


# ============================================================
# ROTATION-TOLERANT INNER SQUARE DETECTION
# ============================================================

def find_inner_square(contours, hierarchy, best_part):
    if hierarchy is None:
        return None

    hierarchy_data = hierarchy[0]
    part_contour = contours[best_part]
    part_area = cv2.contourArea(part_contour)

    px, py, pw, ph = cv2.boundingRect(part_contour)

    if part_area <= 0 or pw <= 0 or ph <= 0:
        return None

    # Use the rotated outer rectangle's center.
    outer_rect = cv2.minAreaRect(part_contour)
    part_cx, part_cy = outer_rect[0]

    outer_side_a, outer_side_b = outer_rect[1]

    if min(outer_side_a, outer_side_b) <= 0:
        return None

    candidates = []

    for i, contour in enumerate(contours):
        if hierarchy_data[i][3] != best_part:
            continue

        area = cv2.contourArea(contour)

        if area <= 0 or area / part_area < MIN_HOLE_AREA_RATIO:
            continue

        inner_rect = cv2.minAreaRect(contour)
        (hole_cx, hole_cy), (rw, rh), _ = inner_rect

        if rw <= 0 or rh <= 0:
            continue

        # Test the rotated dimensions, not the axis-aligned box.
        if min(rw, rh) < MIN_INNER_DIMENSION:
            continue

        aspect = min(rw, rh) / max(rw, rh)

        if aspect < MIN_SQUARE_RATIO:
            continue

        if (
            rw / max(outer_side_a, outer_side_b) < 0.20 or
            rh / max(outer_side_a, outer_side_b) < 0.20
        ):
            continue

        # Center offset measured in image pixels relative to
        # the outer rotated rectangle's size.
        offset_limit = MAX_CENTER_OFFSET_RATIO * max(
            outer_side_a, outer_side_b
        )

        if (
            abs(hole_cx - part_cx) > offset_limit or
            abs(hole_cy - part_cy) > offset_limit
        ):
            continue

        # Confirm that the candidate is actually inside the
        # outer contour rather than just inside its bounding box.
        test_point = (float(hole_cx), float(hole_cy))

        if cv2.pointPolygonTest(
            part_contour, test_point, False
        ) < 0:
            continue

        candidates.append((area, contour))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)

    hole_contour = candidates[0][1]

    inner_rect = cv2.minAreaRect(hole_contour)
    (cx, cy), (rw, rh), _ = inner_rect

    return (
        hole_contour,
        inner_rect,
        float(cx),
        float(cy),
        float(rw),
        float(rh)
    )


# ============================================================
# ROTATION-AWARE CYLINDER DETECTION
# ============================================================

def detect_white_cylinder(hsv, inner_contour):
    """
    Searches for a light-colored cylinder inside the rotated
    inner opening. The mask is clipped to the rotated polygon.
    """
    image_h, image_w = hsv.shape[:2]

    inner_rect = cv2.minAreaRect(inner_contour)
    corners = cv2.boxPoints(inner_rect).astype(np.float32)

    center = corners.mean(axis=0)

    # Shrink the search polygon inward to avoid the red walls.
    inset_scale = 1.0 - 2.0 * CYLINDER_INSET_RATIO

    if inset_scale <= 0:
        return None

    inset_corners = (
        center + (corners - center) * inset_scale
    )

    inset_corners_int = np.round(
        inset_corners
    ).astype(np.int32)

    polygon_mask = np.zeros(
        (image_h, image_w), dtype=np.uint8
    )

    cv2.fillConvexPoly(
        polygon_mask, inset_corners_int, 255
    )

    light_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_MIN_BRIGHTNESS]),
        np.array([180, WHITE_MAX_SATURATION, 255])
    )

    light_mask = cv2.bitwise_and(
        light_mask, polygon_mask
    )

    light_mask = cv2.morphologyEx(
        light_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    light_mask = cv2.morphologyEx(
        light_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2
    )

    contours, _ = cv2.findContours(
        light_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    _, (rw, rh), _ = inner_rect

    min_area = max(
        CYLINDER_MIN_AREA,
        int(rw * rh * MIN_CYLINDER_AREA_RATIO)
    )

    best_contour = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = (
            4 * np.pi * area / (perimeter * perimeter)
        )

        if circularity < MIN_CIRCULARITY:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if radius < MIN_CYLINDER_RADIUS:
            continue

        # Confirm the detected center lies inside the opening.
        if cv2.pointPolygonTest(
            inner_contour,
            (float(cx), float(cy)),
            False
        ) < 0:
            continue

        if area > best_area:
            best_area = area
            best_contour = contour

    return best_contour


# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame(frame):
    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, min(X1, frame_w))
    x2 = max(0, min(X2, frame_w))
    y1 = max(0, min(Y1, frame_h))
    y2 = max(0, min(Y2, frame_h))

    if x2 <= x1 or y2 <= y1:
        update_measurements(camera="ERROR")
        return frame

    # Crop the inspection region, as in your original script.
    roi = frame[y1:y2, x1:x2].copy()
    display = roi.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    current = empty_measurements()

    # --------------------------------------------------------
    # FIND THE RED OUTER PART
    # --------------------------------------------------------

    red_mask = detect_red_part(hsv)

    contours, hierarchy = cv2.findContours(
        red_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_part = None
    best_area = 0

    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # Only top-level contours are candidate outer parts.
            if hierarchy[0][i][3] != -1:
                continue

            area = cv2.contourArea(contour)

            if area >= MIN_PART_AREA and area > best_area:
                best_area = area
                best_part = i

    if best_part is None:
        cv2.putText(
            display,
            "NO PART DETECTED",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED,
            2,
            cv2.LINE_AA
        )

        update_measurements(**current)
        return display

    part_contour = contours[best_part]
    current["part_detected"] = True

    draw_rotated_outline(
        display, part_contour, YELLOW, 2
    )

    # --------------------------------------------------------
    # FIND INNER SQUARE
    # --------------------------------------------------------

    inner_result = find_inner_square(
        contours, hierarchy, best_part
    )

    if inner_result is None:
        cv2.putText(
            display,
            "INNER SQUARE NOT FOUND",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            RED,
            2,
            cv2.LINE_AA
        )

        update_measurements(**current)
        return display

    (
        hole_contour,
        inner_rect,
        inner_cx,
        inner_cy,
        inner_rw,
        inner_rh
    ) = inner_result

    current["inner_square_detected"] = True

    inner_corners = cv2.boxPoints(
        inner_rect
    ).astype(np.int32)

    cv2.polylines(
        display,
        [inner_corners],
        True,
        GREEN,
        2,
        cv2.LINE_AA
    )

    # Rotation-aware inner dimensions.
    length_mm = max(inner_rw, inner_rh) * MM_PER_PIXEL
    width_mm = min(inner_rw, inner_rh) * MM_PER_PIXEL
    inner_area_mm2 = length_mm * width_mm

    current["length"] = round(length_mm, 2)
    current["width"] = round(width_mm, 2)
    current["inner_area"] = round(inner_area_mm2, 2)

    # Draw a dimension line along one edge and its opposite edge.
    cv2.line(
        display,
        tuple(inner_corners[0]),
        tuple(inner_corners[1]),
        RED,
        2,
        cv2.LINE_AA
    )

    cv2.line(
        display,
        tuple(inner_corners[1]),
        tuple(inner_corners[2]),
        PURPLE,
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # WALL THICKNESS
    # --------------------------------------------------------

    wall_measurements = draw_wall_measurements(
        display,
        part_contour,
        hole_contour
    )

    if wall_measurements:
        current["top_wall"] = round(
            wall_measurements["Top"]["thickness_mm"], 2
        )
        current["bottom_wall"] = round(
            wall_measurements["Bottom"]["thickness_mm"], 2
        )
        current["left_wall"] = round(
            wall_measurements["Left"]["thickness_mm"], 2
        )
        current["right_wall"] = round(
            wall_measurements["Right"]["thickness_mm"], 2
        )

    # --------------------------------------------------------
    # CYLINDER DETECTION
    # --------------------------------------------------------

    cylinder_contour = detect_white_cylinder(
        hsv, hole_contour
    )

    if cylinder_contour is not None:
        (cyl_x, cyl_y), cyl_radius = cv2.minEnclosingCircle(
            cylinder_contour
        )

        cylinder_diameter_mm = (
            2 * cyl_radius * MM_PER_PIXEL
        )

        cylinder_area_mm2 = (
            np.pi * (cylinder_diameter_mm / 2) ** 2
        )

        current["cylinder_detected"] = True
        current["cylinder_diameter"] = round(
            cylinder_diameter_mm, 2
        )
        current["cylinder_area"] = round(
            cylinder_area_mm2, 2
        )

        # Original fit-gap formula and threshold preserved.
        fit_gap = inner_area_mm2 - cylinder_area_mm2

        current["fit_gap"] = round(fit_gap, 2)
        current["fit_state"] = (
            "PASS"
            if 0 <= fit_gap <= MAX_FIT_GAP_MM2
            else "FAIL"
        )

        center = (
            int(round(cyl_x)),
            int(round(cyl_y))
        )

        cv2.circle(
            display,
            center,
            int(round(cyl_radius)),
            CYAN,
            2,
            cv2.LINE_AA
        )

        # Draw the cylinder diameter through its center.
        dx = int(round(cyl_radius))
        cv2.line(
            display,
            (center[0] - dx, center[1]),
            (center[0] + dx, center[1]),
            PURPLE,
            2,
            cv2.LINE_AA
        )



    else:
        current["fit_state"] = "NO CYLINDER"

    update_measurements(**current)
    return display


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_worker():
    global latest_jpeg

    cap = cv2.VideoCapture(CAMERA_NUMBER)

    if not cap.isOpened():
        update_measurements(camera="ERROR")
        print("ERROR: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera started.")

    while True:
        ret, frame = cap.read()

        if not ret:
            update_measurements(camera="ERROR")
            time.sleep(0.1)
            continue

        try:
            display = process_frame(frame)

            success, encoded = cv2.imencode(
                ".jpg",
                display,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            if success:
                with data_lock:
                    latest_jpeg = encoded.tobytes()

        except Exception as exc:
            print("Frame processing error:", exc)
            time.sleep(0.05)

    cap.release()


# ============================================================
# LIVE VIDEO STREAM
# ============================================================

def generate_video():
    while True:
        with data_lock:
            frame_bytes = latest_jpeg

        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

        time.sleep(0.01)


# ============================================================
# WEB DASHBOARD
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">


<style>
:root {
    color-scheme: dark;
    --background: #0b0e13;
    --surface: #131923;
    --surface2: #192230;
    --border: #283342;
    --text: #edf2f7;
    --muted: #8e9bac;
    --green: #36d399;
    --red: #ff646e;
    --yellow: #f4c95d;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    background: var(--background);
    color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
    background: #10151d;
}

h1 {
    margin: 0;
    font-size: 20px;
}

.subtitle {
    margin-top: 5px;
    color: var(--muted);
    font-size: 12px;
}

.live-status {
    display: flex;
    align-items: center;
    gap: 9px;
    color: var(--muted);
    font-size: 12px;
}

.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--yellow);
}

main {
    max-width: 1500px;
    margin: auto;
    padding: 25px 28px 35px;
}

.layout {
    display: grid;
    grid-template-columns: minmax(0, 1.8fr) minmax(300px, 1fr);
    gap: 18px;
    align-items: start;
}

.card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    overflow: hidden;
}

.card-header {
    padding: 15px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
}

.card-title {
    font-size: 13px;
    font-weight: 600;
}

.tag {
    font-size: 11px;
    color: var(--green);
    background: #18342c;
    border-radius: 5px;
    padding: 5px 8px;
}

.camera-wrap {
    background: #050608;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 220px;
}

/* Change max-width here to shrink the live camera feed. */
.camera-wrap img {
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
}

.camera-footer {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 11px;
}

.right-column {
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.result-banner {
    padding: 19px;
    border-bottom: 1px solid var(--border);
}

.result-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
}

.result-value {
    margin-top: 9px;
    font-size: 25px;
    font-weight: 700;
}

.pass { color: var(--green); }
.fail { color: var(--red); }
.wait { color: var(--yellow); }

.measure-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    padding: 14px;
}

.metric {
    min-width: 0;
    padding: 14px;
    border: 1px solid var(--border);
    background: var(--surface2);
    border-radius: 8px;
}

.metric-label {
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 10px;
}

.metric-value {
    font-size: 21px;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.unit {
    font-size: 11px;
    font-weight: 400;
    color: var(--muted);
    margin-left: 3px;
}

.wall-list { padding: 4px 18px 12px; }

.wall-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #222c39;
    font-size: 12px;
}

.wall-row:last-child { border-bottom: none; }
.wall-label { color: var(--muted); }
.wall-value { font-weight: 600; }

.footer {
    color: var(--muted);
    font-size: 11px;
    text-align: center;
    margin-top: 23px;
}

@media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    main { padding: 18px 14px; }
    header { padding: 16px; }
}

@media (max-width: 480px) {
    .result-value { font-size: 22px; }
}
</style>
</head>

<body>
<header>
    <div>


    </div>

    <div class="live-status">
        <span class="dot" id="statusDot"></span>
        <span id="connectionStatus">Connecting</span>
    </div>
</header>

<main>
    <div class="layout">

        <section class="card">
            <div class="card-header">
                <span class="card-title">Camera inspection</span>
                <span class="tag">LIVE FEED</span>
            </div>

            <div class="camera-wrap">
                <img src="/video_feed" alt="Live inspection camera">
            </div>

            <div class="camera-footer">
                <span>Camera 0</span>
                <span id="partStatus">Waiting for part</span>
            </div>
        </section>

        <div class="right-column">

            <section class="card">
                <div class="result-banner">
                    <div class="result-label">Inspection result</div>
                    <div class="result-value wait" id="fitState">
                        WAITING
                    </div>
                </div>

                <div class="measure-grid">
                    <div class="metric">
                        <div class="metric-label">Length</div>
                        <div class="metric-value">
                            <span id="length">--</span>
                            <span class="unit">mm</span>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label">Width</div>
                        <div class="metric-value">
                            <span id="width">--</span>
                            <span class="unit">mm</span>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label">Inner area</div>
                        <div class="metric-value">
                            <span id="innerArea">--</span>
                            <span class="unit">mm²</span>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label">Fit gap</div>
                        <div class="metric-value">
                            <span id="fitGap">--</span>
                            <span class="unit">mm²</span>
                        </div>
                    </div>
                </div>
            </section>

            <section class="card">
                <div class="card-header">
                    <span class="card-title">Wall thickness</span>
                    <span class="tag">4 SIDES</span>
                </div>

                <div class="wall-list">
                    <div class="wall-row">
                        <span class="wall-label">Top wall</span>
                        <span class="wall-value" id="topWall">-- mm</span>
                    </div>
                    <div class="wall-row">
                        <span class="wall-label">Bottom wall</span>
                        <span class="wall-value" id="bottomWall">-- mm</span>
                    </div>
                    <div class="wall-row">
                        <span class="wall-label">Left wall</span>
                        <span class="wall-value" id="leftWall">-- mm</span>
                    </div>
                    <div class="wall-row">
                        <span class="wall-label">Right wall</span>
                        <span class="wall-value" id="rightWall">-- mm</span>
                    </div>
                </div>
            </section>

            <section class="card">
                <div class="card-header">
                    <span class="card-title">Cylinder inspection</span>
                </div>

                <div class="measure-grid">
                    <div class="metric">
                        <div class="metric-label">Detection</div>
                        <div class="metric-value" id="cylinderStatus"
                             style="font-size:17px">
                            Waiting
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label">Diameter</div>
                        <div class="metric-value">
                            <span id="cylinderDiameter">--</span>
                            <span class="unit">mm</span>
                        </div>
                    </div>
                </div>
            </section>

        </div>
    </div>

    <div class="footer">
        Live vision processing · Calibration:
        <span id="calibration">0.36339648 mm/pixel</span>
        · <span id="lastUpdate">Waiting for camera</span>
    </div>
</main>

<script>
function fmt(value, decimals = 2) {
    if (value === null || value === undefined) return "--";
    return Number(value).toFixed(decimals);
}

function setText(id, value) {
    document.getElementById(id).textContent = value;
}

async function refreshMeasurements() {
    try {
        const response = await fetch("/measurements", {
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error("Measurement request failed");
        }

        const d = await response.json();

        setText(
            "connectionStatus",
            d.camera === "LIVE" ? "Camera connected" : "Camera unavailable"
        );

        document.getElementById("statusDot").style.background =
            d.camera === "LIVE" ? "var(--green)" : "var(--red)";

        setText(
            "partStatus",
            d.part_detected
                ? (d.inner_square_detected
                    ? "Part detected · Inner square found"
                    : "Part detected · Inner square missing")
                : "No part detected"
        );

        setText("length", fmt(d.length));
        setText("width", fmt(d.width));
        setText("innerArea", fmt(d.inner_area, 1));
        setText("fitGap", fmt(d.fit_gap, 1));

        setText("topWall",
            d.top_wall === null ? "-- mm" : fmt(d.top_wall) + " mm");
        setText("bottomWall",
            d.bottom_wall === null ? "-- mm" : fmt(d.bottom_wall) + " mm");
        setText("leftWall",
            d.left_wall === null ? "-- mm" : fmt(d.left_wall) + " mm");
        setText("rightWall",
            d.right_wall === null ? "-- mm" : fmt(d.right_wall) + " mm");

        setText("cylinderDiameter", fmt(d.cylinder_diameter));

        setText(
            "cylinderStatus",
            d.cylinder_detected ? "Detected" : "Not detected"
        );

        const fit = document.getElementById("fitState");
        fit.textContent = d.fit_state;
        fit.className = "result-value " +
            (d.fit_state === "PASS" ? "pass" :
             d.fit_state === "FAIL" ? "fail" : "wait");

        setText(
            "lastUpdate",
            "Updated " + new Date(d.updated * 1000).toLocaleTimeString()
        );

    } catch (error) {
        setText("connectionStatus", "Reconnecting");
        document.getElementById("statusDot").style.background =
            "var(--red)";
    }
}

refreshMeasurements();
setInterval(refreshMeasurements, 250);
</script>
</body>
</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache"
        }
    )


@app.route("/measurements")
def get_measurements():
    with data_lock:
        data = measurements.copy()

    return jsonify(data)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    camera_thread.start()

    print()
    print("=" * 55)
    print("MANUFACTURING INSPECTION DASHBOARD")
    print("Open your browser at:")
    print("http://127.0.0.1:5000")
    print("Press CTRL+C in this terminal to stop.")
    print("=" * 55)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )

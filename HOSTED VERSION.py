
import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, jsonify, render_template_string

# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_NUMBER = 0

# Calibration - PRESERVED
MM_PER_PIXEL = 0.36339648

# Camera inspection area
X1, Y1 = 150, 70
X2, Y2 = 650, 570

# Red part detection
MIN_PART_AREA = 1000

# Inner square detection
MIN_HOLE_AREA_RATIO = 0.10
MIN_SQUARE_RATIO = 0.60
MAX_CENTER_OFFSET_RATIO = 0.30
MIN_INNER_DIMENSION = 20

# White cylinder detection
WHITE_MIN_BRIGHTNESS = 100
WHITE_MAX_SATURATION = 150
MIN_CYLINDER_AREA_RATIO = 0.08
MIN_CIRCULARITY = 0.40
MIN_CYLINDER_RADIUS = 5
CYLINDER_INSET_RATIO = 0.06
CYLINDER_MIN_AREA = 30

# Display colors (BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLUE = (255, 120, 0)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
DARK_PANEL = (25, 25, 25)

# ============================================================
# SHARED CAMERA AND MEASUREMENT DATA
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


def reset_measurements():
    update_measurements(
        part_detected=False,
        inner_square_detected=False,
        length=None,
        width=None,
        inner_area=None,
        top_wall=None,
        bottom_wall=None,
        left_wall=None,
        right_wall=None,
        cylinder_detected=False,
        cylinder_diameter=None,
        cylinder_area=None,
        fit_gap=None,
        fit_state="WAITING"
    )


# ============================================================
# WALL THICKNESS MEASUREMENT
# ============================================================

def measure_four_walls(outer_contour, inner_contour):

    ox, oy, ow, oh = cv2.boundingRect(outer_contour)
    ix, iy, iw, ih = cv2.boundingRect(inner_contour)

    if ow <= 0 or oh <= 0 or iw <= 0 or ih <= 0:
        return None

    if (
        ix <= ox or
        iy <= oy or
        ix + iw >= ox + ow or
        iy + ih >= oy + oh
    ):
        return None

    walls_px = {
        "Top": iy - oy,
        "Bottom": (oy + oh) - (iy + ih),
        "Left": ix - ox,
        "Right": (ox + ow) - (ix + iw)
    }

    results = {}

    for label, thickness_px in walls_px.items():

        if thickness_px <= 0:
            return None

        results[label] = {
            "thickness_px": float(thickness_px),
            "thickness_mm": thickness_px * MM_PER_PIXEL
        }

    return results


def draw_wall_measurements(display, outer_contour, inner_contour):

    result = measure_four_walls(
        outer_contour,
        inner_contour
    )

    if result is None:
        return None

    ox, oy, ow, oh = cv2.boundingRect(outer_contour)
    ix, iy, iw, ih = cv2.boundingRect(inner_contour)

    cv2.rectangle(
        display, (ox, oy), (ox + ow, oy + oh), BLUE, 2
    )

    cv2.rectangle(
        display, (ix, iy), (ix + iw, iy + ih), GREEN, 2
    )

    outer_cx = ox + ow // 2
    inner_cy = iy + ih // 2

    # Measurement lines
    cv2.line(
        display, (outer_cx, oy), (outer_cx, iy),
        YELLOW, 2, cv2.LINE_AA
    )

    cv2.line(
        display, (outer_cx, iy + ih),
        (outer_cx, oy + oh), YELLOW, 2, cv2.LINE_AA
    )

    cv2.line(
        display, (ox, inner_cy), (ix, inner_cy),
        YELLOW, 2, cv2.LINE_AA
    )

    cv2.line(
        display, (ix + iw, inner_cy),
        (ox + ow, inner_cy), YELLOW, 2, cv2.LINE_AA
    )

    h, w = display.shape[:2]

    labels = [
        ("Top", max(5, outer_cx - 45), max(18, oy - 8)),
        ("Bottom", max(5, outer_cx - 45),
         min(h - 5, oy + oh + 18)),
        ("Left", max(5, ox - 90), max(18, inner_cy - 8)),
        ("Right", min(w - 120, ox + ow + 5),
         max(18, inner_cy - 8))
    ]

    for label, x, y in labels:

        text = (
            f"{label}: "
            f"{result[label]['thickness_mm']:.2f} mm"
        )

        cv2.putText(
            display, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42, YELLOW, 1, cv2.LINE_AA
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
# INNER SQUARE DETECTION
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

    candidates = []

    for i, contour in enumerate(contours):

        if hierarchy_data[i][3] != best_part:
            continue

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        hx, hy, hw, hh = cv2.boundingRect(contour)

        if hw < MIN_INNER_DIMENSION or hh < MIN_INNER_DIMENSION:
            continue

        if area / part_area < MIN_HOLE_AREA_RATIO:
            continue

        aspect = min(hw, hh) / max(hw, hh)

        if aspect < MIN_SQUARE_RATIO:
            continue

        if hw / pw < 0.25 or hh / ph < 0.25:
            continue

        if (
            hx <= px or
            hy <= py or
            hx + hw >= px + pw or
            hy + hh >= py + ph
        ):
            continue

        part_cx = px + pw / 2
        part_cy = py + ph / 2

        hole_cx = hx + hw / 2
        hole_cy = hy + hh / 2

        if abs(hole_cx - part_cx) / pw > MAX_CENTER_OFFSET_RATIO:
            continue

        if abs(hole_cy - part_cy) / ph > MAX_CENTER_OFFSET_RATIO:
            continue

        candidates.append(
            (area, i, hx, hy, hw, hh)
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)

    _, index, hx, hy, hw, hh = candidates[0]

    return contours[index], hx, hy, hw, hh


# ============================================================
# CYLINDER DETECTION
# ============================================================

def detect_white_cylinder(hsv, hx, hy, hw, hh):

    if hw <= 0 or hh <= 0:
        return None

    image_h, image_w = hsv.shape[:2]

    inset_x = max(3, int(hw * CYLINDER_INSET_RATIO))
    inset_y = max(3, int(hh * CYLINDER_INSET_RATIO))

    sx1 = max(0, hx + inset_x)
    sy1 = max(0, hy + inset_y)
    sx2 = min(image_w, hx + hw - inset_x)
    sy2 = min(image_h, hy + hh - inset_y)

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    # White/light gray mask
    light_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_MIN_BRIGHTNESS]),
        np.array([180, WHITE_MAX_SATURATION, 255])
    )

    # Limit search to the inner square
    roi_mask = np.zeros(
        (image_h, image_w),
        dtype=np.uint8
    )

    roi_mask[sy1:sy2, sx1:sx2] = 255

    light_mask = cv2.bitwise_and(light_mask, roi_mask)

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

    min_area = max(
        CYLINDER_MIN_AREA,
        int(hw * hh * MIN_CYLINDER_AREA_RATIO)
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

        if not (sx1 <= cx <= sx2 and sy1 <= cy <= sy2):
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
        return frame

    roi = frame[y1:y2, x1:x2].copy()
    display = roi.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Every frame starts with fresh measurements.
    current = {
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

    # --------------------------------------------------------
    # FIND RED PART
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

            if hierarchy[0][i][3] != -1:
                continue

            area = cv2.contourArea(contour)

            if area >= MIN_PART_AREA and area > best_area:
                best_area = area
                best_part = i

    # --------------------------------------------------------
    # PROCESS CURRENT PART
    # --------------------------------------------------------

    if best_part is None:

        cv2.putText(
            display, "NO PART DETECTED",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, RED, 2, cv2.LINE_AA
        )

        update_measurements(**current)
        return display

    part_contour = contours[best_part]

    px, py, pw, ph = cv2.boundingRect(part_contour)

    current["part_detected"] = True

    cv2.rectangle(
        display,
        (px, py),
        (px + pw, py + ph),
        YELLOW,
        2
    )

    # --------------------------------------------------------
    # INNER SQUARE
    # --------------------------------------------------------

    inner_result = find_inner_square(
        contours, hierarchy, best_part
    )

    if inner_result is None:

        cv2.putText(
            display, "INNER SQUARE NOT FOUND",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, RED, 2, cv2.LINE_AA
        )

        update_measurements(**current)
        return display

    hole_contour, hx, hy, hw, hh = inner_result

    current["inner_square_detected"] = True

    cv2.rectangle(
        display,
        (hx, hy),
        (hx + hw, hy + hh),
        GREEN,
        2
    )

    # Original dimension formulas
    length_mm = hw * MM_PER_PIXEL
    width_mm = hh * MM_PER_PIXEL
    inner_area_mm2 = length_mm * width_mm

    current["length"] = round(length_mm, 2)
    current["width"] = round(width_mm, 2)
    current["inner_area"] = round(inner_area_mm2, 2)

    center_x = hx + hw // 2
    center_y = hy + hh // 2

    cv2.line(
        display, (hx, center_y), (hx + hw, center_y),
        RED, 1
    )

    cv2.line(
        display, (center_x, hy), (center_x, hy + hh),
        RED, 1
    )

    # Wall measurements
    wall_measurements = draw_wall_measurements(
        display, part_contour, hole_contour
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
    # CYLINDER MEASUREMENTS
    # --------------------------------------------------------

    cylinder_contour = detect_white_cylinder(
        hsv, hx, hy, hw, hh
    )

    cylinder_diameter_mm = None
    cylinder_area_mm2 = 0

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

        # Preserve original fit-gap formula and threshold
        fit_gap = inner_area_mm2 - cylinder_area_mm2

        fit_state = (
            "PASS"
            if 0 <= fit_gap <= 160
            else "FAIL"
        )

        current["fit_gap"] = round(fit_gap, 2)
        current["fit_state"] = fit_state

        cyl_center = (
            int(round(cyl_x)),
            int(round(cyl_y))
        )

        radius_draw = int(round(cyl_radius))

        cv2.circle(
            display, cyl_center, radius_draw,
            CYAN, 2, cv2.LINE_AA
        )

        cv2.line(
            display,
            (int(cyl_x - cyl_radius), int(cyl_y)),
            (int(cyl_x + cyl_radius), int(cyl_y)),
            PURPLE, 2, cv2.LINE_AA
        )



    else:
        current["fit_state"] = "NO CYLINDER"

    # Publish fresh measurements for the browser.
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
# VIDEO STREAM
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
    --blue: #70a7ff;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--background);
    color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif;
}

header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
    background: #10151d;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-icon {
    width: 39px;
    height: 39px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    color: var(--green);
    background: #19332d;
    font-size: 22px;
}

h1 {
    margin: 0;
    font-size: 19px;
    font-weight: 650;
    letter-spacing: -0.3px;
}

.subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: 4px;
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
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 10px #36d39955;
}

main {
    max-width: 1500px;
    margin: auto;
    padding: 25px 28px 35px;
}

.section-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

h2 {
    font-size: 15px;
    margin: 0;
    font-weight: 600;
}

.section-note {
    color: var(--muted);
    font-size: 12px;
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
    padding: 16px 18px;
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
    min-height: 280px;
}

.camera-wrap img {
    width: 100%;
    height: auto;
    display: block;
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

.result-value.pass {
    color: var(--green);
}

.result-value.fail {
    color: var(--red);
}

.result-value.wait {
    color: var(--yellow);
}

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
    letter-spacing: -0.5px;
    overflow-wrap: anywhere;
}

.unit {
    font-size: 11px;
    font-weight: 400;
    color: var(--muted);
    margin-left: 3px;
}

.wall-list {
    padding: 4px 18px 12px;
}

.wall-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #222c39;
    font-size: 12px;
}

.wall-row:last-child {
    border-bottom: none;
}

.wall-label {
    color: var(--muted);
}

.wall-value {
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

.footer {
    color: var(--muted);
    font-size: 11px;
    text-align: center;
    margin-top: 23px;
}

@media (max-width: 900px) {
    .layout {
        grid-template-columns: 1fr;
    }

    main {
        padding: 18px 14px;
    }

    header {
        padding: 16px;
    }
}

@media (max-width: 480px) {
    .section-note {
        display: none;
    }

    .result-value {
        font-size: 22px;
    }
}
</style>
</head>

<body>

<header>


    <div class="live-status">
        <span class="dot" id="statusDot"></span>
        <span id="connectionStatus">Connecting</span>
    </div>
</header>

<main>

    <div class="section-heading">
        <span class="section-note">Measurements refresh automatically</span>
    </div>

    <div class="layout">

        <!-- LIVE CAMERA -->
        <section class="card">
            <div class="card-header">
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

            <!-- INSPECTION RESULT -->
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

            <!-- WALL THICKNESS -->
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

            <!-- CYLINDER -->
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
        <span id="calibration">0.24654524 mm/pixel</span>
        · <span id="lastUpdate">Waiting for camera</span>
    </div>

</main>

<script>
function setText(id, value, suffix = "") {
    const el = document.getElementById(id);
    el.textContent =
        value === null || value === undefined ? "--" : value + suffix;
}

function formatValue(value) {
    return value === null || value === undefined
        ? "--"
        : Number(value).toFixed(2);
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

        document.getElementById("connectionStatus").textContent =
            d.camera === "LIVE" ? "Camera connected" : "Camera unavailable";

        document.getElementById("statusDot").style.background =
            d.camera === "LIVE" ? "var(--green)" : "var(--red)";

        document.getElementById("partStatus").textContent =
            d.part_detected
                ? (d.inner_square_detected
                    ? "Part detected · Inner square found"
                    : "Part detected · Inner square missing")
                : "No part detected";

        setText("length", formatValue(d.length));
        setText("width", formatValue(d.width));
        setText("innerArea",
            d.inner_area === null ? "--" : Number(d.inner_area).toFixed(1));
        setText("fitGap",
            d.fit_gap === null ? "--" : Number(d.fit_gap).toFixed(1));

        setText("topWall",
            d.top_wall === null ? "--" : formatValue(d.top_wall), " mm");
        setText("bottomWall",
            d.bottom_wall === null ? "--" : formatValue(d.bottom_wall), " mm");
        setText("leftWall",
            d.left_wall === null ? "--" : formatValue(d.left_wall), " mm");
        setText("rightWall",
            d.right_wall === null ? "--" : formatValue(d.right_wall), " mm");

        setText("cylinderDiameter",
            d.cylinder_diameter === null
                ? "--"
                : formatValue(d.cylinder_diameter));

        document.getElementById("cylinderStatus").textContent =
            d.cylinder_detected ? "Detected" : "Not detected";

        const fit = document.getElementById("fitState");
        fit.textContent = d.fit_state;

        fit.className = "result-value " +
            (d.fit_state === "PASS" ? "pass" :
             d.fit_state === "FAIL" ? "fail" : "wait");

        document.getElementById("lastUpdate").textContent =
            "Updated " + new Date(d.updated * 1000).toLocaleTimeString();

    } catch (error) {
        document.getElementById("connectionStatus").textContent =
            "Reconnecting";

        document.getElementById("statusDot").style.background =
            "var(--red)";
    }
}

// Poll measurements without reloading the page.
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
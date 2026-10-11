

import cv2
import numpy as np
import threading
from flask import Flask, Response

# SETT UP

CAMERA_NUMBER = 0

# Calibration

MM_PER_PIXEL = 0.24654524

# Camera inspection area

X1, Y1 = 150, 70

X2, Y2 = 650, 570

# Red part detection

MIN_PART_AREA = 1000

# Inner square detection

MIN_HOLE_AREA_RATIO = 0.10

MIN_SQUARE_RATIO = 0.60

MAX_CENTER_OFFSET_RATIO = 0.30

# White cylinder detection

WHITE_MIN_BRIGHTNESS = 100

WHITE_MAX_SATURATION = 150

MIN_CYLINDER_AREA_RATIO = 0.08

MIN_CIRCULARITY = 0.40

MIN_CYLINDER_RADIUS = 5

# Colors: BGR

RED = (0, 0, 255)

GREEN = (0, 255, 0)

YELLOW = (0, 255, 255)

CYAN = (255, 255, 0)

WHITE = (255, 255, 255)

BLUE = (255, 120, 0)

ORANGE = (0, 165, 255)

PURPLE = (255, 0, 255)

# WEB STREAMING
# Open http://127.0.0.1:5000 in your browser while this script is running.
app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

@app.route("/")
def index():
    return """<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Part Inspection</title>
</head>
<body style="margin:0; padding:16px; background:#222; color:white; font-family:Arial,sans-serif; text-align:center;">
    <h2>Inner Square and Wall Inspection</h2>
    <img src="/video_feed" alt="Live inspection feed" style="width:100%; max-width:1000px; height:auto; border:1px solid #555;">
    <p>Live camera inspection. Close the Python window or press Q there to stop.</p>
</body>
</html>"""

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                # Briefly yield control while the camera initializes.
                import time
                time.sleep(0.03)
                continue

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def run_flask():
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )

# WALL THICKNESS MEASUREMENT

def measure_four_walls(outer_contour, inner_contour):

    """

    Measure top, bottom, left and right wall thicknesses.

    Uses independently detected outer and inner contours.

    Designed for a nearly straight-on camera view.

    """

    ox, oy, ow, oh = cv2.boundingRect(outer_contour)

    ix, iy, iw, ih = cv2.boundingRect(inner_contour)

    if ow <= 0 or oh <= 0 or iw <= 0 or ih <= 0:

        return None

    # Ensure the inner contour is inside the outer contour.

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

    """Draw wall measurement lines and labels."""

    measurements = measure_four_walls(

        outer_contour,

        inner_contour

    )

    if measurements is None:

        cv2.putText(

            display,

            "WALL MEASUREMENT FAILED",

            (10, display.shape[0] - 15),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            ORANGE,

            2,

            cv2.LINE_AA

        )

        return None

    ox, oy, ow, oh = cv2.boundingRect(outer_contour)

    ix, iy, iw, ih = cv2.boundingRect(inner_contour)

    # Draw the measured outer and inner boundaries.

    cv2.rectangle(

        display,

        (ox, oy),

        (ox + ow, oy + oh),

        BLUE,

        2

    )

    cv2.rectangle(

        display,

        (ix, iy),

        (ix + iw, iy + ih),

        GREEN,

        2

    )

    # Centers of each wall.

    outer_cx = ox + ow // 2

    inner_cx = ix + iw // 2

    outer_cy = oy + oh // 2

    inner_cy = iy + ih // 2

    # Top wall measurement line.

    cv2.line(

        display,

        (outer_cx, oy),

        (outer_cx, iy),

        YELLOW,

        2,

        cv2.LINE_AA

    )

    # Bottom wall measurement line.

    cv2.line(

        display,

        (outer_cx, iy + ih),

        (outer_cx, oy + oh),

        YELLOW,

        2,

        cv2.LINE_AA

    )

    # Left wall measurement line.

    cv2.line(

        display,

        (ox, inner_cy),

        (ix, inner_cy),

        YELLOW,

        2,

        cv2.LINE_AA

    )

    # Right wall measurement line.

    cv2.line(

        display,

        (ix + iw, inner_cy),

        (ox + ow, inner_cy),

        YELLOW,

        2,

        cv2.LINE_AA

    )

    h, w = display.shape[:2]

    labels = [

        (

            "Top",

            max(5, outer_cx - 45),

            max(18, oy - 8)

        ),

        (

            "Bottom",

            max(5, outer_cx - 45),

            min(h - 5, oy + oh + 18)

        ),

        (

            "Left",

            max(5, ox - 90),

            max(18, inner_cy - 8)

        ),

        (

            "Right",

            min(w - 120, ox + ow + 5),

            max(18, inner_cy - 8)

        )

    ]

    for label, x, y in labels:

        text = (

            f"{label}: "

            f"{measurements[label]['thickness_mm']:.2f} mm"

        )

        cv2.putText(

            display,

            text,

            (x, y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.42,

            YELLOW,

            1,

            cv2.LINE_AA

        )

    return measurements

# ============================================================

# CAMERA

# ============================================================

cap = cv2.VideoCapture(CAMERA_NUMBER)

if not cap.isOpened():

    print("ERROR: Could not open camera.")

    raise SystemExit

print("Camera started.")

print("Press Q to quit.")

# Start the browser stream in the background.
threading.Thread(target=run_flask, daemon=True).start()
print("Browser view: http://127.0.0.1:5000")

# MAIN LOOP

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Could not read camera.")

            break

        # CROP INSPECTION AREA

        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, min(X1, frame_w))

        x2 = max(0, min(X2, frame_w))

        y1 = max(0, min(Y1, frame_h))

        y2 = max(0, min(Y2, frame_h))

        if x2 <= x1 or y2 <= y1:

            print("ERROR: Inspection area is outside camera frame.")

            break

        roi = frame[y1:y2, x1:x2].copy()

        display = roi.copy()

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # DETECT RED PART

        lower_red1 = np.array([0, 70, 70])

        upper_red1 = np.array([15, 255, 255])

        lower_red2 = np.array([165, 70, 70])

        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(

            hsv,

            lower_red1,

            upper_red1

        )

        red_mask2 = cv2.inRange(

            hsv,

            lower_red2,

            upper_red2

        )

        red_mask = cv2.bitwise_or(

            red_mask1,

            red_mask2

        )

        kernel = np.ones((5, 5), np.uint8)

        red_mask = cv2.morphologyEx(

            red_mask,

            cv2.MORPH_OPEN,

            kernel

        )

        red_mask = cv2.morphologyEx(

            red_mask,

            cv2.MORPH_CLOSE,

            kernel

        )

        contours, hierarchy = cv2.findContours(

            red_mask,

            cv2.RETR_CCOMP,

            cv2.CHAIN_APPROX_SIMPLE

        )

        best_part = None

        best_part_area = 0

        if hierarchy is not None:

            hierarchy_data = hierarchy[0]

            for i, contour in enumerate(contours):

                if hierarchy_data[i][3] != -1:

                    continue

                area = cv2.contourArea(contour)

                if area < MIN_PART_AREA:

                    continue

                if area > best_part_area:

                    best_part_area = area

                    best_part = i

        # PROCESS DETECTED PART

        if best_part is not None:

            part_contour = contours[best_part]

            px, py, pw, ph = cv2.boundingRect(

                part_contour

            )

            cv2.rectangle(

                display,

                (px, py),

                (px + pw, py + ph),

                YELLOW,

                2

            )

            part_area = cv2.contourArea(part_contour)

            hole_candidates = []

            # FIND INNER SQUARE CONTOUR

            hierarchy_data = hierarchy[0]

            for i, contour in enumerate(contours):

                if hierarchy_data[i][3] != best_part:

                    continue

                area = cv2.contourArea(contour)

                if area <= 0 or part_area <= 0:

                    continue

                hx0, hy0, hw0, hh0 = cv2.boundingRect(

                    contour

                )

                if hw0 < 20 or hh0 < 20:

                    continue

                area_ratio = area / part_area

                if area_ratio < MIN_HOLE_AREA_RATIO:

                    continue

                aspect_ratio = (

                    min(hw0, hh0) / max(hw0, hh0)

                )

                if aspect_ratio < MIN_SQUARE_RATIO:

                    continue

                if hw0 / pw < 0.25:

                    continue

                if hh0 / ph < 0.25:

                    continue

                part_cx = px + pw / 2

                part_cy = py + ph / 2

                hole_cx = hx0 + hw0 / 2

                hole_cy = hy0 + hh0 / 2

                offset_x = abs(

                    hole_cx - part_cx

                ) / pw

                offset_y = abs(

                    hole_cy - part_cy

                ) / ph

                if offset_x > MAX_CENTER_OFFSET_RATIO:

                    continue

                if offset_y > MAX_CENTER_OFFSET_RATIO:

                    continue

                hole_candidates.append(

                    (

                        area,

                        i,

                        hx0,

                        hy0,

                        hw0,

                        hh0

                    )

                )

            # SELECT INNER SQUARE

            hole_contour = None

            if hole_candidates:

                hole_candidates.sort(

                    key=lambda item: item[0],

                    reverse=True

                )

                (

                    hole_area,

                    hole_index,

                    hx,

                    hy,

                    hw,

                    hh

                ) = hole_candidates[0]

                hole_contour = contours[hole_index]

            else:

                # Fallback for approximate inner-square dimensions.

                # Wall measurements will not use this estimate.

                hw = int(pw * 0.60)

                hh = int(ph * 0.66)

                hx = int(px + (pw - hw) / 2)

                hy = int(py + (ph - hh) / 2)

            # INNER SQUARE MEASUREMENTS

            if hw >= 20 and hh >= 20:

                cv2.rectangle(

                    display,

                    (hx, hy),

                    (hx + hw, hy + hh),

                    GREEN,

                    2

                )

                length_mm = hw * MM_PER_PIXEL

                width_mm = hh * MM_PER_PIXEL

                inner_area_mm2 = (

                    length_mm * width_mm

                )

                hole_center_x = hx + hw // 2

                hole_center_y = hy + hh // 2

                cv2.line(

                    display,

                    (hx, hole_center_y),

                    (hx + hw, hole_center_y),

                    RED,

                    1

                )

                cv2.line(

                    display,

                    (hole_center_x, hy),

                    (hole_center_x, hy + hh),

                    RED,

                    1

                )

                cv2.putText(

                    display,

                    "L",

                    (hx + 3, max(15, hy - 5)),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    RED,

                    1,

                    cv2.LINE_AA

                )

                cv2.putText(

                    display,

                    "W",

                    (

                        min(

                            display.shape[1] - 18,

                            hx + hw + 14

                        ),

                        hole_center_y + 5

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    RED,

                    1,

                    cv2.LINE_AA

                )

                # WALL THICKNESS MEASUREMENTS

                wall_measurements = None

                if hole_contour is not None:

                    wall_measurements = draw_wall_measurements(

                        display,

                        part_contour,

                        hole_contour

                    )

                else:

                    cv2.putText(

                        display,

                        "WALLS: INNER CONTOUR NOT FOUND",

                        (10, display.shape[0] - 15),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.45,

                        ORANGE,

                        2,

                        cv2.LINE_AA

                    )

                # DETECT WHITE / LIGHT-GRAY CYLINDER

                lower_light = np.array([

                    0,

                    0,

                    WHITE_MIN_BRIGHTNESS

                ])

                upper_light = np.array([

                    180,

                    WHITE_MAX_SATURATION,

                    255

                ])

                light_mask = cv2.inRange(

                    hsv,

                    lower_light,

                    upper_light

                )

                hole_mask = np.zeros(

                    light_mask.shape,

                    dtype=np.uint8

                )

                inset = 7

                inner_x1 = hx + inset

                inner_y1 = hy + inset

                inner_x2 = hx + hw - inset

                inner_y2 = hy + hh - inset

                if (

                    inner_x2 > inner_x1

                    and inner_y2 > inner_y1

                ):

                    cv2.rectangle(

                        hole_mask,

                        (inner_x1, inner_y1),

                        (inner_x2, inner_y2),

                        255,

                        -1

                    )

                light_in_hole = cv2.bitwise_and(

                    light_mask,

                    hole_mask

                )

                light_in_hole = cv2.morphologyEx(

                    light_in_hole,

                    cv2.MORPH_OPEN,

                    np.ones((3, 3), np.uint8)

                )

                light_in_hole = cv2.morphologyEx(

                    light_in_hole,

                    cv2.MORPH_CLOSE,

                    np.ones((5, 5), np.uint8),

                    iterations=2

                )

                cylinder_contours, _ = cv2.findContours(

                    light_in_hole,

                    cv2.RETR_EXTERNAL,

                    cv2.CHAIN_APPROX_SIMPLE

                )

                cylinder_contour = None

                largest_cylinder_area = 0

                min_cylinder_area = max(

                    30,

                    int(

                        hw * hh *

                        MIN_CYLINDER_AREA_RATIO

                    )

                )

                for contour in cylinder_contours:

                    area = cv2.contourArea(contour)

                    if area < min_cylinder_area:

                        continue

                    perimeter = cv2.arcLength(

                        contour,

                        True

                    )

                    if perimeter <= 0:

                        continue

                    circularity = (

                        4 * np.pi * area /

                        (perimeter * perimeter)

                    )

                    if circularity < MIN_CIRCULARITY:

                        continue

                    (cx, cy), radius = (

                        cv2.minEnclosingCircle(contour)

                    )

                    if radius < MIN_CYLINDER_RADIUS:

                        continue

                    if area > largest_cylinder_area:

                        largest_cylinder_area = area

                        cylinder_contour = contour

                # CYLINDER MEASUREMENT

                cylinder_diameter_mm = None

                cylinder_area_mm2 = 0

                if cylinder_contour is not None:

                    (cyl_x, cyl_y), cyl_radius = (

                        cv2.minEnclosingCircle(

                            cylinder_contour

                        )

                    )

                    cylinder_diameter_mm = (

                        2 * cyl_radius * MM_PER_PIXEL

                    )

                    cylinder_area_mm2 = (

                        np.pi *

                        (cylinder_diameter_mm / 2) ** 2

                    )

                    cyl_center = (

                        int(round(cyl_x)),

                        int(round(cyl_y))

                    )

                    radius_draw = int(round(cyl_radius))

                    cv2.circle(

                        display,

                        cyl_center,

                        radius_draw,

                        CYAN,

                        2,

                        cv2.LINE_AA

                    )

                    cv2.line(

                        display,

                        (

                            int(round(cyl_x - cyl_radius)),

                            int(round(cyl_y))

                        ),

                        (

                            int(round(cyl_x + cyl_radius)),

                            int(round(cyl_y))

                        ),

                        PURPLE,

                        2,

                        cv2.LINE_AA

                    )

                # MEASUREMENT PANEL

                if cylinder_diameter_mm is not None:

                    fit_gap = (

                        inner_area_mm2 -

                        cylinder_area_mm2

                    )

                    fit_state = (

                        "PASS"

                        if 0 <= fit_gap <= 140

                        else "FAIL"

                    )

                    cylinder_text = (

                        f"Cylinder Diameter: "

                        f"{cylinder_diameter_mm:.2f} mm"

                    )

                else:

                    fit_gap = 0

                    fit_state = "NOT MEASURED"

                    cylinder_text = "Cylinder: NOT DETECTED"

                panel_lines = [

                    ("Measurements:", GREEN, 0.52, 2),

                    (

                        f"Length: {length_mm:.2f} mm",

                        WHITE, 0.43, 1

                    ),

                    (

                        f"Width: {width_mm:.2f} mm",

                        WHITE, 0.43, 1

                    )

                ]

                for label in (

                    "Top",

                    "Bottom",

                    "Left",

                    "Right"

                ):

                    if (

                        wall_measurements is not None

                        and label in wall_measurements

                    ):

                        value = (

                            wall_measurements[label][

                                "thickness_mm"

                            ]

                        )

                        panel_lines.append(

                            (

                                f"{label} wall: {value:.2f} mm",

                                YELLOW, 0.40, 1

                            )

                        )

                    else:

                        panel_lines.append(

                            (

                                f"{label} wall: NOT MEASURED",

                                ORANGE, 0.36, 1

                            )

                        )

                panel_lines.extend([

                    (

                        f"Inner area: {inner_area_mm2:.1f} mm2",

                        WHITE, 0.38, 1

                    ),

                    (

                        f"Fit gap: {fit_gap:.1f} mm2",

                        WHITE, 0.38, 1

                    ),

                    (

                        f"Fit: {fit_state}",

                        WHITE, 0.38, 1

                    ),

                    (

                        cylinder_text,

                        WHITE, 0.36, 1

                    )

                ])

                panel_x = max(

                    5,

                    display.shape[1] - 180

                )

                panel_y = 10

                panel_w = min(

                    210,

                    display.shape[1] - panel_x - 5

                )

                line_spacing = 22

                panel_h = (

                    18 +

                    len(panel_lines) * line_spacing

                )

                cv2.rectangle(

                    display,

                    (panel_x, panel_y),

                    (

                        panel_x + panel_w,

                        panel_y + panel_h

                    ),

                    (30, 30, 30),

                    -1

                )

                cv2.rectangle(

                    display,

                    (panel_x, panel_y),

                    (

                        panel_x + panel_w,

                        panel_y + panel_h

                    ),

                    GREEN,

                    2

                )

                for i, (

                    text,

                    color,

                    scale,

                    thickness

                ) in enumerate(panel_lines):

                    cv2.putText(

                        display,

                        text,

                        (

                            panel_x + 7,

                            panel_y + 22 +

                            i * line_spacing

                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        scale,

                        color,

                        thickness,

                        cv2.LINE_AA

                    )

            else:

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

        else:

            cv2.putText(

                display,

                "NO RED PART DETECTED",

                (15, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                RED,

                2,

                cv2.LINE_AA

            )

        # DISPLAY

        # Publish the annotated image to the browser.
        with frame_lock:
            latest_frame = display.copy()

        cv2.imshow(

            "Inner Square and Wall Inspection",

            display

        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

finally:

    cap.release()

    cv2.destroyAllWindows()

    print("Camera stopped.")

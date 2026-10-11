
import cv2

# ==========================================
# CALIBRATION SETTINGS
# ==========================================

CAMERA_NUMBER = 0

KNOWN_LENGTH_MM = 25.29
KNOWN_WIDTH_MM = 25.22

WINDOW_NAME = "Camera Calibration"

# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_NUMBER)

if not cap.isOpened():
    print("Could not open camera.")
    print("Try changing CAMERA_NUMBER to 1.")
    exit()

print("CAMERA CALIBRATION")
print("1. Place your known object flat in view.")
print("2. Press SPACE to freeze the image.")
print("3. Drag a rectangle around the object's edges.")
print("4. Press ENTER to calculate calibration.")
print("5. Press C to retry or Q to quit.")

# ==========================================
# MOUSE SELECTION
# ==========================================

selecting = False
start_point = None
end_point = None
rectangle = None
frozen_frame = None
select_mode = False


def mouse_callback(event, x, y, flags, param):
    global selecting, start_point, end_point, rectangle

    if not select_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        start_point = (x, y)
        end_point = (x, y)
        rectangle = None

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        end_point = (x, y)

        x1, y1 = start_point
        x2, y2 = end_point

        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if width > 5 and height > 5:
            rectangle = (left, top, width, height)


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    if not select_mode:
        ret, frame = cap.read()

        if not ret:
            print("Camera read error.")
            break

        display = frame.copy()

        cv2.putText(
            display,
            "SPACE = Freeze image",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, display)

    else:
        display = frozen_frame.copy()

        if rectangle is not None:
            x, y, w, h = rectangle

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Selected: {w} x {h} pixels",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        elif selecting and start_point and end_point:
            cv2.rectangle(
                display,
                start_point,
                end_point,
                (0, 255, 255),
                2
            )

        cv2.putText(
            display,
            "ENTER = Calibrate | C = Retry | Q = Quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1) & 0xFF

    # Freeze current camera image
    if not select_mode and key == ord(' '):
        frozen_frame = frame.copy()
        rectangle = None
        select_mode = True

    # Retry with a fresh image
    elif select_mode and key == ord('c'):
        select_mode = False
        rectangle = None

    # Calculate calibration
    elif select_mode and key in (13, 10):

        if rectangle is None:
            print("Select the object first.")
            continue

        x, y, pixel_width, pixel_height = rectangle

        # Calculate horizontal and vertical scales
        mm_per_pixel_x = KNOWN_LENGTH_MM / pixel_width
        mm_per_pixel_y = KNOWN_WIDTH_MM / pixel_height

        # Average both measurements
        mm_per_pixel = (
            mm_per_pixel_x + mm_per_pixel_y
        ) / 2.0

        # Calculate disagreement between scales
        disagreement = (
            abs(mm_per_pixel_x - mm_per_pixel_y)
            / mm_per_pixel * 100
        )

        print("\n========== CALIBRATION RESULTS ==========")
        print(f"Object width:  {pixel_width} pixels")
        print(f"Object height: {pixel_height} pixels")

        print(f"Horizontal scale: {mm_per_pixel_x:.10f} mm/pixel")
        print(f"Vertical scale:   {mm_per_pixel_y:.10f} mm/pixel")

        print("\nUse this value in your measurement program:")
        print(f"MM_PER_PIXEL = {mm_per_pixel:.10f}")

        print(f"\nScale disagreement: {disagreement:.2f}%")

        if disagreement > 3:
            print("WARNING: The scales differ by over 3%.")
            print("Check your selection, camera angle, and object orientation.")

        print("==========================================\n")

        # Keep the result on screen
        result = frozen_frame.copy()

        cv2.rectangle(
            result,
            (x, y),
            (x + pixel_width, y + pixel_height),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            f"MM_PER_PIXEL = {mm_per_pixel:.8f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(WINDOW_NAME, result)

        print("Press C to recalibrate or Q to quit.")

        while True:
            k = cv2.waitKey(0) & 0xFF

            if k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

            if k == ord('c'):
                select_mode = False
                rectangle = None
                break

    elif key == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Calibration program closed.")
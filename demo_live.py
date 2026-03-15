import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

TAU_CONF = 0.50
TAU_AREA = 0.01


model = YOLO("best.pt")

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Erreur : webcam non détectée")
    exit()

os.makedirs("logs", exist_ok=True)
log_path = f"logs/demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(log_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["timestamp", "scenario_id", "max_conf", "area_ratio",
                 "tau_conf", "tau_area", "decision", "notes"])

scenario_id = 1
last_logged_decision = None

print("=" * 50)
print("  DEMO LIVE — Drone Landing Detection")
print(f"  τ_conf={TAU_CONF} | τ_area={TAU_AREA}")
print("=" * 50)
print("Touches : [S] logger scénario | [Q] quitter")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_area = frame.shape[0] * frame.shape[1]

    results = model.predict(frame, conf=0.01, iou=0.6, verbose=False)
    r = results[0]

    max_conf = 0.0
    area_ratio = 0.0

    if len(r.boxes) > 0:
        best_idx = r.boxes.conf.argmax()
        max_conf = float(r.boxes.conf[best_idx])
        box = r.boxes.xywh[best_idx]
        area_ratio = float((box[2] * box[3]) / img_area)

        x1, y1, x2, y2 = map(int, r.boxes.xyxy[best_idx])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    is_safe = max_conf >= TAU_CONF and area_ratio >= TAU_AREA
    decision = "SAFE" if is_safe else "NOT SAFE"
    color = (0, 200, 0) if is_safe else (0, 0, 220)

    cv2.rectangle(frame, (0, 0), (420, 110), (30, 30, 30), -1)
    cv2.putText(frame, f"Decision : {decision}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Conf     : {max_conf:.3f}  (min {TAU_CONF})",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"Area     : {area_ratio*100:.1f}%  (min {TAU_AREA*100:.1f}%)",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"Scenario : {scenario_id}  |  [S] log  [Q] quit",
                (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Drone Landing Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        notes = input(f"  → Notes pour scénario {scenario_id} : ")
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            scenario_id,
            round(max_conf, 4),
            round(area_ratio, 4),
            TAU_CONF,
            TAU_AREA,
            decision,
            notes
        ])
        csv_file.flush()
        print(f"  ✅ Scénario {scenario_id} loggé → {decision} | conf={max_conf:.3f} | area={area_ratio*100:.1f}%")
        scenario_id += 1

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"\nLog sauvegardé : {log_path}")
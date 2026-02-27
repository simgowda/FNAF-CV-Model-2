import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import mediapipe as mp
from torchvision import transforms
from torchvision.models import resnet18
from collections import deque, Counter

MODEL_PATH = "gesture_resnet18.pt"
CAMERA_INDEX = 0
SMOOTHING_WINDOW = 8

CONF_THRESHOLD = 0.80     
UNKNOWN_LABEL = "Unknown"

# ---- load model ----
ckpt = torch.load(MODEL_PATH, map_location="cpu")
CLASSES = ckpt["classes"]
IMG_SIZE = ckpt["img_size"]

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ---- transforms ----
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# ---- MediaPipe ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def crop_from_landmarks(frame, hand_landmarks, pad=40):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, w)
    y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, h)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

pred_hist = deque(maxlen=SMOOTHING_WINDOW)

cap = cv2.VideoCapture(CAMERA_INDEX)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    label_text = "No hand"
    conf_text = ""

    if res.multi_hand_landmarks:
        hlm = res.multi_hand_landmarks[0]
        crop, (x1, y1, x2, y2) = crop_from_landmarks(frame, hlm, pad=50)

        if crop.size > 0:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            x = tfm(pil).unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                idx = int(torch.argmax(probs))
                conf = float(probs[idx])

            # confidence gate
            if conf >= CONF_THRESHOLD:
                pred_hist.append(idx)
                smooth_idx = Counter(pred_hist).most_common(1)[0][0]
                label_text = CLASSES[smooth_idx]
            else:
                label_text = UNKNOWN_LABEL

            conf_text = f"{conf:.2f}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(frame, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
    if conf_text:
        cv2.putText(frame, f"conf: {conf_text}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Gesture Demo", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

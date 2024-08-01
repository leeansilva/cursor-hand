import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

# inicializando MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Configuración de pyautogui
screen_width, screen_height = pyautogui.size()
prev_x, prev_y = 0, 0
umbral_movimiento = 12  
scaling_factor = 2.0  
click_threshold = 100  
click_hold_threshold = 0.2 
click_hold = False
hold_start_time = 0

def move_mouse(hand_landmarks):
    global prev_x, prev_y
    
    # Obtener las coordenadas del índice
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Convertir las coordenadas a la resolución de la pantalla e invertir el eje X
    x = screen_width - int(index_finger_tip.x * screen_width * scaling_factor)
    y = int(index_finger_tip.y * screen_height * scaling_factor)
    
    # Limitar los valores de x e y a la resolución de la pantalla
    x = max(0, min(screen_width, x))
    y = max(0, min(screen_height, y))
    
    # Calcular la distancia de movimiento
    dist_x = abs(x - prev_x)
    dist_y = abs(y - prev_y)
    
    if dist_x > umbral_movimiento or dist_y > umbral_movimiento:
        pyautogui.moveTo(x, y)
        
        prev_x, prev_y = x, y

def detect_click(hand_landmarks):
    global click_hold, hold_start_time
    
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    
    # Calcular la distancia entre el dedo meñique y el pulgar
    dist = ((pinky_tip.x - thumb_tip.x) ** 2 + (pinky_tip.y - thumb_tip.y) ** 2) ** 0.5 * screen_width
    
    current_time = time.time()
    
    if dist < click_threshold:
        if not click_hold:
            hold_start_time = current_time
            pyautogui.mouseDown()
            click_hold = False
    else:
        pyautogui.mouseUp()
        
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            move_mouse(hand_landmarks)
            detect_click(hand_landmarks)

    cv2.imshow('Hand Gesture Recognition', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

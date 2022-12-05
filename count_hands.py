# Importando librerias necesarias
import cv2
import mediapipe as mp

# Iniciar sistema de detección
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Iniciar camara
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    #Comprobar Entrada
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Convertir Imagen a RGB
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Dibujar puntos de detección
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Establezca inicialmente el conteo de dedos en 0
    fingerCount = 0

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        # Obtener el índice de la mano para comprobar la etiqueta (izquierda o derecha)
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        # Establece la variable para mantener las posiciones de los puntos de referencia (x e y)
        handLandmarks = []

        # Rellenar la lista con las posiciones X y Y de cada punto de referencia
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])

        # Condiciones de prueba para cada dedo: El recuento se incrementa si el dedo se 
        #       se considera levantado.
        # Pulgar: La posición TIP x debe ser mayor o menor que la posición IP x, 
        #       dependiendo de la etiqueta de la mano.
        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
          fingerCount = fingerCount+1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          fingerCount = fingerCount+1

        # Otros dedos: La posición TIP y debe ser inferior a la posición PIP y, 
        #       ya que el origen de la imagen está en la esquina superior izquierda.
        if handLandmarks[8][1] < handLandmarks[6][1]:       #Dedo Indice
          fingerCount = fingerCount+1
        if handLandmarks[12][1] < handLandmarks[10][1]:     #Dedo medio
          fingerCount = fingerCount+1
        if handLandmarks[16][1] < handLandmarks[14][1]:     #Dedo anular
          fingerCount = fingerCount+1
        if handLandmarks[20][1] < handLandmarks[18][1]:     #Meñique
          fingerCount = fingerCount+1

        # Dibujar puntos de referencia de la mano 
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Muestra el conteo de dedos
    cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Mostrar imagen
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
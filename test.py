import cv2
import mediapipe as mp
import numpy as np
import pygame

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
min_tracking_confidence=0.5)
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
bg = pygame.image.load("40714741.jpeg")
bg = pygame.transform.scale(bg, (WIDTH * 2, HEIGHT * 2))
cap = cv2.VideoCapture(0)
while cap.isOpened():
        success, frame = cap.read()
        if not success:
    
            break
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(frame_rgb)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263] 
        head_x = (left_eye.x + right_eye.x) / 2
        head_y = (left_eye.y + right_eye.y) / 2
        offset_x = int((head_x - 0.5) * 400)
        offset_y = int((head_y - 0.5) * 200)
        screen.fill((0, 0, 0))
        screen.blit(bg, (-WIDTH // 2 + offset_x, -HEIGHT // 2 + offset_y))
        pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()
clock.tick(30)
cap.release()
pygame.quit()
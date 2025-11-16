#!/usr/bin/env python3
"""
Hand-controlled Snake using Mediapipe + OpenCV.

Controls:
 - Move your index fingertip to move the snake head
 - p : pause / resume
 - r : restart after game over
 - q or ESC : quit
"""
import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque

# -------- CONFIG --------
CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720

# smoothing
POS_BUFFER_LEN = 4
SMOOTHING_ALPHA = 0.25

# game params
INITIAL_LENGTH = 8          # initial snake length (segments)
SEGMENT_SIZE = 20           # visual segment radius
MOVE_STEP = 16              # head moves toward pointer by this many pixels per tick (controls speed)
FOOD_RADIUS = 12
EAT_DISTANCE = 22           # distance threshold to consider food eaten
SELF_COLLISION_DIST = 18    # distance threshold for self-collision
MAX_FOOD_TRIES = 200

FPS = 30

# Mediapipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.5)

# state
pos_buffer = deque(maxlen=POS_BUFFER_LEN)
smoothed_x = None
smoothed_y = None

def norm_to_frame(x_norm, y_norm, frame_w, frame_h):
    """Normalized mediapipe coords (0..1) -> frame coords; flip x for mirror effect."""
    sx = (1.0 - x_norm) * frame_w
    sy = y_norm * frame_h
    sx = int(max(0, min(frame_w-1, sx)))
    sy = int(max(0, min(frame_h-1, sy)))
    return sx, sy

def spawn_food(snake_points, frame_w, frame_h):
    """Spawn food not on the snake body. Return (x,y)."""
    tries = 0
    while tries < MAX_FOOD_TRIES:
        x = random.randint(FOOD_RADIUS+10, frame_w-FOOD_RADIUS-10)
        y = random.randint(FOOD_RADIUS+10, frame_h-FOOD_RADIUS-10)
        collision = False
        for px, py in snake_points:
            if np.hypot(px - x, py - y) < (SEGMENT_SIZE + FOOD_RADIUS + 8):
                collision = True
                break
        if not collision:
            return (x, y)
        tries += 1
    # fallback
    return (random.randint(50, frame_w-50), random.randint(50, frame_h-50))

def distance(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def game_loop():
    global smoothed_x, smoothed_y, pos_buffer

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # snake represented by deque of points (head at index 0)
    snake = deque()
    center = (FRAME_W//2, FRAME_H//2)
    for i in range(INITIAL_LENGTH):
        snake.append((center[0] - i*SEGMENT_SIZE, center[1]))

    food = spawn_food(snake, FRAME_W, FRAME_H)
    score = 0
    paused = False
    game_over = False

    last_time = time.time()
    wait_time = 1.0 / FPS

    info_text = "Point with your index finger to move. p=Pause r=Restart q/ESC=Quit"

    while True:
        now = time.time()
        dt = now - last_time
        if dt < wait_time:
            time.sleep(wait_time - dt)
        last_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pointer = None
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = hand.landmark[8]  # index fingertip
            px, py = norm_to_frame(lm.x, lm.y, w, h)
            # smoothing
            pos_buffer.append((px, py))
            bx = int(np.median([p[0] for p in pos_buffer]))
            by = int(np.median([p[1] for p in pos_buffer]))
            if smoothed_x is None:
                smoothed_x, smoothed_y = bx, by
            else:
                smoothed_x = SMOOTHING_ALPHA * bx + (1 - SMOOTHING_ALPHA) * smoothed_x
                smoothed_y = SMOOTHING_ALPHA * by + (1 - SMOOTHING_ALPHA) * smoothed_y
            pointer = (int(smoothed_x), int(smoothed_y))
            # optional draw pointer
            cv2.circle(frame, pointer, 10, (200, 80, 255), -1)

        # Game logic (only advance when not paused and not game_over)
        if not paused and not game_over and pointer is not None:
            head = snake[0]
            # compute vector toward pointer
            vec_x = pointer[0] - head[0]
            vec_y = pointer[1] - head[1]
            dist = np.hypot(vec_x, vec_y)
            if dist > 1:
                # move head by a fixed step toward pointer (clamp to dist)
                step = min(MOVE_STEP, dist)
                new_head = (int(head[0] + (vec_x/dist)*step), int(head[1] + (vec_y/dist)*step))
            else:
                new_head = head

            # insert new head
            snake.appendleft(new_head)
            # unless eating, pop tail to keep length
            if distance(new_head, food) < EAT_DISTANCE:
                score += 1
                # grow by adding extra segments (don't pop)
                # spawn new food
                food = spawn_food(snake, w, h)
            else:
                snake.pop()

            # check self-collision: head too close to any body point after some initial segments
            for i, pt in enumerate(list(snake)[5:]):  # ignore immediate neck segments
                if distance(new_head, pt) < SELF_COLLISION_DIST:
                    game_over = True
                    print("Game Over: collided with self")
                    break

            # check wall collision? (optional) - we allow wrap-around or you can enable game_over on wall hit
            # Example: if head outside frame -> game over
            hx, hy = new_head
            if hx < 0 or hy < 0 or hx >= w or hy >= h:
                # wrap-around instead of game over:
                hx = max(0, min(w-1, hx))
                hy = max(0, min(h-1, hy))
                new_head = (hx, hy)
                snake[0] = new_head
                # Or uncomment next line to make wall collision end the game:
                # game_over = True

        # draw food
        cv2.circle(frame, food, FOOD_RADIUS, (0,180,0), -1)
        cv2.putText(frame, "Food", (food[0]+FOOD_RADIUS+4, food[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,0), 1)

        # draw snake segments
        pts = list(snake)
        for i, (x,y) in enumerate(pts):
            # color gradient: head bright red, tail darker
            alpha = 1.0 - (i / max(1, len(pts)-1))
            col = (int(50 + 205*alpha), int(50), int(220 * (1-alpha)))
            cv2.circle(frame, (x,y), SEGMENT_SIZE, col, -1)
            # small outline for readability
            cv2.circle(frame, (x,y), SEGMENT_SIZE, (0,0,0), 1)

        # HUD
        cv2.putText(frame, f"Score: {score}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, info_text, (12, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if paused:
            cv2.putText(frame, "PAUSED - press p to resume", (frame.shape[1]//2 - 200, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,200), 2)

        if game_over:
            cv2.putText(frame, "GAME OVER! Press r to restart", (frame.shape[1]//2 - 240, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
            cv2.putText(frame, f"Final score: {score}", (frame.shape[1]//2 - 140, frame.shape[0]//2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("Hand Snake (point with index finger)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            # restart
            snake = deque()
            for i in range(INITIAL_LENGTH):
                snake.append((center[0] - i*SEGMENT_SIZE, center[1]))
            food = spawn_food(snake, FRAME_W, FRAME_H)
            score = 0
            paused = False
            game_over = False
            smoothed_x = None
            smoothed_y = None
            pos_buffer.clear()
            print("Restarted game")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    game_loop()

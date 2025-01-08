import cv2
import mediapipe as mp
import math
from openni import openni2

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# OpenNI 초기화 (Astra SDK 경로 설정)
openni2.initialize("/path/to/Orbbec/OpenNI2/Redist")
device = openni2.Device.open_any()
depth_stream = device.create_depth_stream()
depth_stream.start()

# Depth 데이터 가져오기
def get_depth(x, y):
    frame = depth_stream.read_frame()
    depth_data = frame.get_buffer_as_uint16()
    width, height = depth_stream.get_video_mode().resolutionX, depth_stream.get_video_mode().resolutionY
    depth_array = depth_data.reshape((height, width))
    if 0 <= x < width and 0 <= y < height:
        return depth_array[y, x]
    return None

# 3D 거리 계산 (피타고라스 정리)
def calculate_3d_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

# 삼각함수로 거리 계산
def calculate_distance_with_trig(d1, d2, d_xy):
    """
    삼각형 법칙:
    c^2 = a^2 + b^2 - 2ab * cos(θ)
    """
    if d1 is None or d2 is None:
        return None
    # cos(θ) 계산
    cos_theta = d_xy / math.sqrt(d1**2 + d2**2)
    if cos_theta > 1 or cos_theta < -1:  # 오차 처리
        cos_theta = max(-1, min(1, cos_theta))
    angle = math.acos(cos_theta)  # θ 계산
    # 피타고라스 확장으로 거리 계산
    d_fingers = math.sqrt(d1**2 + d2**2 - 2 * d1 * d2 * cos_theta)
    return d_fingers, math.degrees(angle)  # 거리와 각도 반환

# 카메라 스트림 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe로 손 인식
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 엄지와 검지 끝점의 2D 좌표
            h, w, _ = frame.shape
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_2d = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_2d = (int(index_tip.x * w), int(index_tip.y * h))

            # Depth 데이터 가져오기
            thumb_z = get_depth(thumb_2d[0], thumb_2d[1])
            index_z = get_depth(index_2d[0], index_2d[1])

            if thumb_z and index_z:
                # Depth 카메라 거리
                d1 = thumb_z
                d2 = index_z

                # 2D 거리 계산
                d_xy = math.sqrt((index_2d[0] - thumb_2d[0])**2 + (index_2d[1] - thumb_2d[1])**2)

                # 삼각함수로 거리 계산
                trig_distance, angle = calculate_distance_with_trig(d1, d2, d_xy)

                # 3D 거리 계산
                thumb_3d = (thumb_2d[0], thumb_2d[1], thumb_z)
                index_3d = (index_2d[0], index_2d[1], index_z)
                distance_3d = calculate_3d_distance(thumb_3d, index_3d)

                # 결과 출력
                print(f"3D Distance (Pythagoras): {distance_3d:.2f}mm")
                print(f"Trig Distance: {trig_distance:.2f}mm, Angle: {angle:.2f} degrees")

                # 화면에 표시
                cv2.circle(image, thumb_2d, 10, (255, 0, 0), -1)
                cv2.circle(image, index_2d, 10, (0, 255, 0), -1)
                cv2.line(image, thumb_2d, index_2d, (255, 255, 255), 2)
                cv2.putText(image, f"{distance_3d:.2f}mm", (index_2d[0], index_2d[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"{trig_distance:.2f}mm (Trig)", (index_2d[0], index_2d[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 화면 출력
    cv2.imshow('3D Finger Distance with Trigonometry', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()

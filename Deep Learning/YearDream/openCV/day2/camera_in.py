import sys
import cv2


# 카메라 열기
cap = cv2.VideoCapture(0)
# cap = VideoCapture('video1.mp4')
if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 카메라 프레임 크기 출력
print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


# 카메라 프레임 처리
num = 0
cnt = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # inversed = ~frame  # 반전

    num += 1

    if num == 30:
        cnt += 1
        filename = 'frame_%04d.jpg' % cnt
        cv2.imwrite(filename, frame)
        num = 0

    cv2.imshow('frame', frame)
    cv2.imshow('inversed', inversed)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()

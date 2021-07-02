import cv2
import os
import time
import mtcnn

images_path = '../test_images/'
model = mtcnn.mtcnn.MTCNN()
image_files = sorted(os.listdir(images_path))
timings = []

for image_name in image_files:
    image = cv2.imread(os.path.join(images_path, image_name))
    initial_time = time.time()
    faces = model.detect_faces(image)
    final_time = time.time()
    timings.append(final_time - initial_time)
    print(faces)
    for face in faces:
        (x, y, w, h) = face['box']
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        image = cv2.circle(image, face['keypoints']['left_eye'], 3, (255, 0, 0), -1)
        image = cv2.circle(image, face['keypoints']['right_eye'], 3, (255, 0, 0), -1)
        image = cv2.circle(image, face['keypoints']['nose'], 3, (255, 0, 0), -1)
        image = cv2.circle(image, face['keypoints']['mouth_left'], 3, (255, 0, 0), -1)
        image = cv2.circle(image, face['keypoints']['mouth_right'], 3, (255, 0, 0), -1)
    cv2.imshow('CNN MTCNN detector: {}'.format(image_name), image)
    cv2.imwrite('image_results/{}_result.jpg'.format(image_name.split('.')[0]), image)
    cv2.waitKey(0)
    cv2.destroyWindow('CNN MTCNN detector: {}'.format(image_name))

cv2.destroyAllWindows()

print(timings)
print(sum(timings))
print(sum(timings)/15)

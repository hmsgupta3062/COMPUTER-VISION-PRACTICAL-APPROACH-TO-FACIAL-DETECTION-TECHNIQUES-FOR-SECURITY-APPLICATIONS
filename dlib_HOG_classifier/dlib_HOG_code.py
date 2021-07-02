import cv2
import dlib
import os
import time
import numpy

images_path = '../test_images/'
predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
detector_model = dlib.get_frontal_face_detector()
predictor_model = dlib.shape_predictor(predictor_path)
image_files = sorted(os.listdir(images_path))
timings = []

for image_name in image_files:
    image = cv2.imread(os.path.join(images_path, image_name))
    initial_time = time.time()
    faces = detector_model(image, 1)
    final_time = time.time()
    timings.append(final_time - initial_time)
    for face in faces:
        face_points = str(face).split()
        (x1, y1, x2, y2) = int(face_points[0][2:-1]), int(face_points[1][:-1]), int(face_points[2][1:-1]), int(face_points[3][:-2])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        landmarks = numpy.matrix([[point.x, point.y] for point in predictor_model(image, face).parts()])
        for point in landmarks:
            x3, y3 = point[0, 0], point[0, 1]
            cv2.circle(image, (x3, y3), 2, (255, 0, 0), -1)
    cv2.imshow('dlib HOG Classifier: {}'.format(image_name), image)
    cv2.imwrite('image_results/{}_result.jpg'.format(image_name.split('.')[0]), image)
    cv2.waitKey(0)
    cv2.destroyWindow('dlib HOG Classifier: {}'.format(image_name))

cv2.destroyAllWindows()

print(timings)
print(sum(timings))
print(sum(timings)/15)

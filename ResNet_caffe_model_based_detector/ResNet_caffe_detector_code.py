import cv2
import numpy
import os
import time

images_path = '../test_images/'
model = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
image_files = sorted(os.listdir(images_path))
timings = []

for image_name in image_files:
    image = cv2.imread(os.path.join(images_path, image_name))
    height, width = image.shape[:2]
    initial_time = time.time()
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (500, 500), (104.0, 177.0, 123.0))
    model.setInput(blob)
    faces = model.forward()
    final_time = time.time()
    timings.append(final_time - initial_time)
    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.6:
            box = faces[0, 0, i, 3:7] * numpy.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imshow('ResNet face detector: {}'.format(image_name), image)
    cv2.imwrite('image_results/{}_result.jpg'.format(image_name.split('.')[0]), image)
    cv2.waitKey(0)
    cv2.destroyWindow('ResNet face detector: {}'.format(image_name))

cv2.destroyAllWindows()

print(timings)
print(sum(timings))
print(sum(timings)/15)

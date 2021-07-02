import cv2
import os
import time

images_path = '../test_images/'
model_path = 'models/haarcascade_frontalface_default.xml'
model = cv2.CascadeClassifier(model_path)
image_files = sorted(os.listdir(images_path))
timings = []

for image_name in image_files:
    image = cv2.imread(os.path.join(images_path, image_name))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    initial_time = time.time()
    faces = model.detectMultiScale(image_gray, 1.25, 4)
    final_time = time.time()
    timings.append(final_time - initial_time)
    for (x, y, w, h) in faces:
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imshow('Haar Cascade Classifier: {}'.format(image_name), image)
    cv2.imwrite('image_results/{}_result.jpg'.format(image_name.split('.')[0]), image)
    cv2.waitKey(0)
    cv2.destroyWindow('Haar Cascade Classifier: {}'.format(image_name))

cv2.destroyAllWindows()

print(timings)
print(sum(timings))
print(sum(timings)/15)

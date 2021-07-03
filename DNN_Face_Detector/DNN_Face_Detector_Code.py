# import the necessary libraries
import cv2
import numpy
import os
import time

# declare the paths of the test dataset and models
images_path = '../test_images/'
prototxt_path = 'models/deploy.prototxt.txt'
model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'

# load the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

image_files = sorted(os.listdir(images_path))  # get the test images name
timings = []  # initialise a list to store time complexities of the model on each and every image
faces_count = []

# initialise a loop to process each and every image once
for image_name in image_files:
    image = cv2.imread(os.path.join(images_path, image_name))  # read the image
    height, width = image.shape[:2]  # store the height and width of the image
    initial_time = time.time()  # record time before detecting faces

    # detect the faces in the image using the face detector model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (500, 500), (104.0, 177.0, 123.0))
    model.setInput(blob)
    faces = model.forward()

    final_time = time.time()  # record time after detecting faces
    timings.append(final_time - initial_time)  # store the time complexity of the image
    count = 0
    # initialise a nested loop to process all the detected faces individually
    for i in range(0, faces.shape[2]):
        # process the each face individually, check for the confidence value,
        # retrieve the coordinates of the bounding boxes and draw them
        confidence = faces[0, 0, i, 2]
        if confidence > 0.6:
            box = faces[0, 0, i, 3:7] * numpy.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            count += 1

    faces_count.append(count)

    cv2.imshow('ResNet face detector: {}'.format(image_name), image)  # display the resultant image
    cv2.imwrite('image_results/{}_result.jpg'.format(image_name.split('.')[0]), image)  # save the resultant image
    cv2.waitKey(0)  # wait for the user to press any key
    cv2.destroyWindow('ResNet face detector: {}'.format(image_name))  # destroy the current image window

cv2.destroyAllWindows()  # destroy all the image windows

print(timings)  # print the time complexities of the images


print(sum(timings))
print(sum(timings) / 15)
print(faces_count)
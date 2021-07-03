# import the necessary libraries
import cv2
import dlib
import os
import time
import numpy

# declare the paths of the test dataset and models
images_path = '../test_images/'
model_path = 'models/mmod_human_face_detector.dat'
predictor_path = 'models/shape_predictor_68_face_landmarks.dat'

# load the models
detector_model = dlib.cnn_face_detection_model_v1(model_path)
predictor_model = dlib.shape_predictor(predictor_path)

image_files = sorted(os.listdir(images_path))  # get the test images name
timings = []  # initialise a list to store time complexities of the model on each and every image
faces_count = []

# initialise a loop to process each and every image once
for image_name in image_files:
    image = cv2.imread(os.path.join(images_path, image_name))  # read the image
    initial_time = time.time()  # record time before detecting faces
    faces = detector_model(image, 1)  # detect the faces in the image using the face detector model
    final_time = time.time()  # record time after detecting faces
    timings.append(final_time - initial_time)  # store the time complexity of the image
    faces_count.append(len(faces))

    # initialise a nested loop to process all the detected faces individually
    for face in faces:
        # process the faces individually, retrieve the coordinates of the bounding boxes and draw them
        face_points = str(face.rect).split()
        (x1, y1, x2, y2) = int(face_points[0][2:-1]), int(face_points[1][:-1]), int(face_points[2][1:-1]), int(
            face_points[3][:-2])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # detect the facial landmarks of the current face using the shape predictor model
        landmarks = numpy.matrix([[point.x, point.y] for point in predictor_model(image, face.rect).parts()])

        # initialise a new nested loop inside the nested loop to process all the facial landmarks
        # of the current face individually
        for point in landmarks:
            # process the facial landmarks individually, retrieve the coordinates of the facial
            # landmarks and mark them
            x3, y3 = point[0, 0], point[0, 1]
            cv2.circle(image, (x3, y3), 2, (255, 0, 0), -1)

    cv2.imshow('dlib CNN Classifier: {}'.format(image_name), image)  # display the resultant image
    cv2.imwrite('image_results/{}_result.jpg'.format(image_name.split('.')[0]), image)  # save the resultant image
    cv2.waitKey(0)  # wait for the user to press any key
    cv2.destroyWindow('dlib CNN Classifier: {}'.format(image_name))  # destroy the current image window

cv2.destroyAllWindows()  # destroy all the image windows

print(timings)  # print the time complexities of the images


print(sum(timings))
print(sum(timings) / 15)
print(faces_count)
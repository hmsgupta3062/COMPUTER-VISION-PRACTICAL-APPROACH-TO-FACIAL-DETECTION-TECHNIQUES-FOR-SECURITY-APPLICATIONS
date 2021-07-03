# import the necessary libraries
import cv2
import os
import time
import mtcnn

# declare the path of the test dataset
images_path = '../test_images/'

# load the model
model = mtcnn.mtcnn.MTCNN()

image_files = sorted(os.listdir(images_path))  # get the test images name
timings = []  # initialise a list to store time complexities of the model on each and every image
faces_count = []

# initialise a loop to process each and every image once
for image_name in image_files:
    image = cv2.imread(os.path.join(images_path, image_name))  # read the image
    initial_time = time.time()  # record time before detecting faces
    faces = model.detect_faces(image)  # detect the faces in the image using the face detector model
    final_time = time.time()  # record time after detecting faces
    timings.append(final_time - initial_time)  # store the time complexity of the image
    faces_count.append(len(faces))

    # initialise a nested loop to process all the detected faces individually
    for face in faces:
        # Retrieve the coordinates of the bounding boxes of the current face and draw them
        (x, y, w, h) = face['box']
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Retrieve the coordinates of the facial landmarks of the current face and draw them
        image = cv2.circle(image, face['keypoints']['left_eye'], 3, (255, 0, 0), -1)
        image = cv2.circle(image, face['keypoints']['right_eye'], 3, (255, 0, 0), -1)
        image = cv2.circle(image, face['keypoints']['nose'], 3, (255, 0, 0), -1)
        image = cv2.circle(image, face['keypoints']['mouth_left'], 3, (255, 0, 0), -1)
        image = cv2.circle(image, face['keypoints']['mouth_right'], 3, (255, 0, 0), -1)

    cv2.imshow('CNN MTCNN detector: {}'.format(image_name), image)  # display the resultant image
    cv2.imwrite('image_results/{}_result.jpg'.format(image_name.split('.')[0]), image)  # save the resultant image
    cv2.waitKey(0)  # wait for the user to press any key
    cv2.destroyWindow('CNN MTCNN detector: {}'.format(image_name))  # destroy the current image window

cv2.destroyAllWindows()  # destroy all the image windows

print(timings)  # print the time complexities of the images


print(sum(timings))
print(sum(timings) / 15)
print(faces_count)
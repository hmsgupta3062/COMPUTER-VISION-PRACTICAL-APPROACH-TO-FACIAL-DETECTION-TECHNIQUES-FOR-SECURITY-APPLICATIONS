# import the necessary libraries
import cv2
import os
import time

# declare the paths of the test dataset and model
images_path = '../test_images/'
model_path = 'models/haarcascade_frontalface_default.xml'

# load the model
model = cv2.CascadeClassifier(model_path)

image_files = sorted(os.listdir(images_path))  # get the test images name
timings = []  # initialise a list to store time complexities of the model on each and every image
faces_count = []

# initialise a loop to process each and every image once
for image_name in image_files:
    image = cv2.imread(os.path.join(images_path, image_name))  # read the image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the image from BGR to the grayscale
    initial_time = time.time()  # record time before detecting faces
    faces = model.detectMultiScale(image_gray, 1.25, 4)  # detect the faces in the image using the face detector model
    final_time = time.time()  # record time after detecting faces
    timings.append(final_time - initial_time)  # store the time complexity of the image
    faces_count.append(len(faces))

    # initialise a nested loop to process all the detected faces individually
    for (x, y, w, h) in faces:
        # process the faces individually, retrieve the coordinates of the bounding boxes and draw them
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow('Haar Cascade Classifier: {}'.format(image_name), image)  # display the resultant image
    cv2.imwrite('image_results/{}_result.jpg'.format(image_name.split('.')[0]), image)  # save the resultant image
    cv2.waitKey(0)  # wait for the user to press any key
    cv2.destroyWindow('Haar Cascade Classifier: {}'.format(image_name))  # destroy the current image window

cv2.destroyAllWindows()  # destroy all the image windows

print(timings)  # print the time complexities of the images


print(sum(timings))
print(sum(timings) / 15)
print(faces_count)
import cv2
import dlib

# get face detector and predictor from dlib
detec = dlib.get_frontal_face_detector()  # used to find faces
predic = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # used to find facial features

# read video
vid = cv2.VideoCapture(0)

while True:
	_, frame = vid.read()
	# grayscale video
	gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

	# find faces (can be more than one)
	faces = detec(gray)

	for face in faces:
		# define area to look for facial features
		x1 = face.left()  # leftmost point
		y1 = face.top()  # topmost point
		x2 = face.right()  # rightmost point
		y2 = face.bottom()  # bottommost point

		# object used to map facial features
		features = predic(image=gray, box=face)

		# loop through facial feature points
		for n in range(0, 68):
			x = features.part(n).x  # x coordinate of point n
			y = features.part(n).y # y coordinate of point n

			# draws point n (usefull for visualization)
			cv2.circle(img=frame, center=(x,y), radius=4, color=(0, 255, 0), thickness=-1)

	# show video (useful for visualization)
	cv2.imshow(winname="Vid", mat=frame)

	# exit when escape is pressed
	if cv2.waitKey(delay=1) == 27:
		break
# release video and objects
vid.release()

# close all windows
cv2.destroyAllWindows()
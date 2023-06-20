import cv2
import dlib
import numpy



# get face detector and predictor from dlib
detec = dlib.get_frontal_face_detector()  # used to find faces
predic = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # used to find facial features


# read video
vid = cv2.VideoCapture(0)

mask1 = []
mask2 = []
mask3 = []
mask4 = []
rEye = []
lEye = []
mouth = []

while True:
	_, frame = vid.read()
	# grayscale video
	gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

	# find faces (can be more than one)
	faces = detec(gray)

	for face in faces:
		mask1 = []
		mask2 = []
		mask3 = []
		mask4 = []
		rEye = []
		lEye = []
		mouth = []
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

			# add main face points to mask1 array
			if n >= 0 and n <= 16:
				mask1.append([x,y])

			# add points to mask arrays 2, 3 and 4 to complete mask
			if n == 0 or n == 15 or n==19:
				mask2.append([x,y])
			if n == 1 or n == 16 or n==24:
				mask3.append([x,y])
			if n == 8 or n == 19 or n==24:
				mask4.append([x,y])

			# add right eye points to rEye array
			if n >= 36 and n <= 41:
				rEye.append([x,y])

			# add left eye points to lEye array
			if n >= 42 and n <= 47:
				lEye.append([x,y])

			# add mouth points to mouth array
			if n >= 60 and n <= 67:
				mouth.append([x,y])

		

		# draw face in white
		cv2.fillPoly(img=frame, pts=[numpy.int32(mask1)], color=(200,200,200))
		cv2.fillPoly(img=frame, pts=[numpy.int32(mask2)], color=(200,200,200))
		cv2.fillPoly(img=frame, pts=[numpy.int32(mask3)], color=(200,200,200))
		cv2.fillPoly(img=frame, pts=[numpy.int32(mask4)], color=(200,200,200))

		# draw right eye in blue
		cv2.fillPoly(img=frame, pts=[numpy.int32(rEye)], color=(255,0,0))

		# draw left eye in blue
		cv2.fillPoly(img=frame, pts=[numpy.int32(lEye)], color=(255,0,0))

		# draw mouth in red
		cv2.fillPoly(img=frame, pts=[numpy.int32(mouth)], color=(0,0,255))

		# display right face distance
		cv2.putText(
			frame,str(features.part(16).x - features.part(27).x),
			(1200,700),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

		# display left face distance
		cv2.putText(
			frame,str(features.part(27).x - features.part(0).x),
			(0,700),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

		# display face rotation
		cv2.putText(
			frame,str(((features.part(27).x - features.part(0).x)-(features.part(16).x - features.part(27).x))/6),
			(600,700),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

		cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

		
		

	# show video (useful for visualization)
	cv2.imshow(winname="Vid", mat=frame)

	# exit when escape is pressed
	if cv2.waitKey(delay=1) == 27:
		break
# release video and objects
vid.release()

# close all windows
cv2.destroyAllWindows()
from imutils import paths
from imtools import Scissors
from keras.models import load_model
import argparse
import json
import cv2


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument(
		'-c',
		'--conf',
		default='config/default.json',
		help='path to the JSON config file'
	)

	args = vars(ap.parse_args())
	conf = json.load(open(args['conf']))

	digit_models = load_model(conf['digit_model'])

	for imagePath in list(paths.list_images(conf["images"])):
		image = cv2.imread(imagePath)
		weather_info = image[1750:1800, 1325:1570, :]
		lunar_info = image[1750:1815, 1570:1635, :]
		datetime_info = image[1750:1800, 2650:3264, :]

		boxes, thresh = Scissors.cut(datetime_info)

		# loop over the started bounding boxes
		for (startX, startY, endX, endY) in boxes:
			char = cv2.resize(thresh[startY:endY, startX:endX], (28, 28), interpolation=cv2.INTER_AREA)
			cv2.imshow("", char)

			char = char.reshape((1, 1, 28, 28))
			prediction = digit_models.predict_classes(char)
			print(prediction)

			cv2.waitKey(0)


if __name__ == '__main__':
	main()

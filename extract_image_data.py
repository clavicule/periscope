from imtools import Scissors
from keras.models import load_model
import argparse
import json
import cv2
import xmltodict
import csv


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

	# load the model recognizing digits
	digit_models = load_model(conf['digit_model'])

	output = []
	with open(conf['xml_tags']) as fd:
		doc = xmltodict.parse(fd.read())

		# get all the tags
		all_tags = {}
		for tag in doc['dataset']['tags']['tag']:
			all_tags[tag['@name']] = 0

		for im in doc['dataset']['images']['image']:
			tags = all_tags.copy()
			image_path = im['@file']
			image = cv2.imread(image_path)

			# hard-code text position for now
			# lunar_info = image[1750:1815, 1570:1635, :]
			temp_info = image[1750:1800, 1300:1400, :]
			datetime_info = image[1750:1800, 2650:3264, :]

			# get temperature
			temp_string = ""
			boxes, thresh = Scissors.cut(temp_info)
			for (startX, startY, endX, endY) in boxes:
				char = cv2.resize(thresh[startY:endY, startX:endX], (28, 28), interpolation=cv2.INTER_AREA)
				char = char.reshape((1, 1, 28, 28))
				prediction = digit_models.predict_classes(char)
				temp_string += str(prediction[0])

			# get date and time
			date_string = ""
			time_string = ""
			boxes, thresh = Scissors.cut(datetime_info)
			for (startX, startY, endX, endY) in boxes:
				char = cv2.resize(thresh[startY:endY, startX:endX], (28, 28), interpolation=cv2.INTER_AREA)
				char = char.reshape((1, 1, 28, 28))
				prediction = digit_models.predict_classes(char)

				if len(date_string) < 8:
					date_string += str(prediction[0])
				else:
					time_string += str(prediction[0])

			# post-process:
			# data = "MM-DD-YYYY"
			# time = "HH:MM:SS"
			date_string = date_string[:2] + "-" + date_string[2:4] + "-" + date_string[4:]
			time_string = time_string[:2] + ":" + time_string[2:4] + ":" + time_string[4:]

			# several labels
			if isinstance(im['box'], list):
				for b in im['box']:
					tags[b['label']] += 1
			else:
				tags[im['box']['label']] += 1

			# get count per tag
			for t in tags:
				if tags[t] > 0:
					path_decomp = image_path.split('/')
					output.append(
						tuple([path_decomp[-1], path_decomp[-2], date_string, time_string, temp_string, t, tags[t]])
					)

	with open(conf['output'], 'w') as out:
		out.write("picture name, folder, date(MM-DD-YYYY), time(HH:MM:SS), temperature, tag, occurrence \n")
		csv_out = csv.writer(out)
		for row in output:
			csv_out.writerow(row)


if __name__ == '__main__':
	main()

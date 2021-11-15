import os
from app import app
from flask import flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from pred import *
from gevent.pywsgi import WSGIServer

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
	return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		# print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)


@app.route('/plate_number/<filename>')
def predict_info(filename):
	# print('display_image filename: ' + filename)
	_, plate = plate_detect(url_for('static', filename='uploads/' + filename))
	# char_list = segment_characters(plate)
	# plate_number = show_results(char_list)
	vehicle_info = get_vehicle_info('AP26BZ1998')
	return render_template('upload.html', vehicle_info=vehicle_info)


if __name__ == "__main__":
	http_server = WSGIServer(('', 5000), app)
	http_server.serve_forever()

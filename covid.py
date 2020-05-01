from flask import Flask, request, render_template, flash
app = Flask(__name__)

from common import get_tensor
from inference import diagnosis_type

@app.route('/', methods=['GET', 'POST'])
def hello_world():
	if request.method == 'GET':
		return render_template('index.html', value='hi')
	if request.method == 'POST':
		print(request.files)
		file = request.files['file']
		image = file.read()
		name = diagnosis_type(image_bytes=image)
		return render_template('result.html', name=name)

if __name__ == '__main__':
	app.run(debug=True)
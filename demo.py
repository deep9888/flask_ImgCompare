from flask import Flask, redirect, url_for, render_template, request
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from scipy.spatial import distance
import os
metric = 'cosine'



model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

IMAGE_SHAPE = (128, 128)

layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
def extract(file):
  file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
  #display(file)

  file = np.stack((file,)*3, axis=-1)

  file = np.array(file)/255.0

  embedding = model.predict(file[np.newaxis, ...])
  #print(embedding)
  vgg16_feature_np = np.array(embedding)
  flattended_feature = vgg16_feature_np.flatten()

  #print(len(flattended_feature))
  #print(flattended_feature)
  #print('-----------')
  return flattended_feature

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
	return render_template('index.html')

@app.route("/", methods=['POST'])
def getValue():
	if request.method == 'POST':
		img1= request.files['file1']
		img2 = request.files['file2']
		image_path_screenshot="images/screenshots/"+img1.filename
		img1.save(image_path_screenshot)
	
		image_path_reference="images/reference image/"+img2.filename
		img2.save(image_path_reference)

		org1 = extract(img1)
		org2 = extract(img2)

		dc = distance.cdist([org1], [org2], metric)[0]
		print(dc)
		if dc< 0.40:
			prediction = "Matched"
		else:
			prediction = "Not Matched"
		print("the distance between Rendered Model Screenshot with it's own Reference Image {}".format(dc))
	

		return render_template('index.html',pred=prediction)
	else:
		prediction=''
		return render_template('index.html',pred=prediction)


# @app.route("/<name>")
# def user(name):
# 	return f"Hello {name}!"

# #redirects to home webpage
# @app.route("/admin")
# def admin():
# 	return redirect(url_for("home"))

if __name__ == '__main__':
	app.run(debug=True)
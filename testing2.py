from flask import Flask, redirect, url_for, render_template, request
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
app = Flask(__name__)

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

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

		image_names = [image_path_screenshot,image_path_reference]
		
		encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
		# # Now we run the clustering algorithm. This function compares images aganist 
		# # all other images and returns a list with the pairs that have the highest 
		# # cosine similarity score
		processed_images = util.paraphrase_mining_embeddings(encoded_image)
		NUM_SIMILAR_IMAGES = 2 

		# # =================
		# # NEAR DUPLICATES
		# # =================
		
		print('Finding near duplicate images...')
		# Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
		# you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
		# A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
		# duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.
		threshold = 0.99
		near_duplicates = [image for image in processed_images if image[0] < threshold]
		print(near_duplicates)
		for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
	 		#print("\nScore: {:.3f}%".format(score * 100))
	 		print("the distance between Rendered Model Screenshot with it's own Reference Image {}".format(score))
	
	 		print(image_path_screenshot)
	 		print(image_path_reference)

		if score<0.89:
			prediction = "Not Matched - - "+ str(score)
		else:
			prediction = "Matched - - " + str(score)
		
		return render_template('index.html',pred=prediction)
	else:
	 	prediction=''
	 	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
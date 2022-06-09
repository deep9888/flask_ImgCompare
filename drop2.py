from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
import urllib.request
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageChops
import glob
import os
 

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
       
# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32') 
app.config.update(
    UPLOADED_PATH= os.path.join(basedir,'uploads'),
    DROPZONE_MAX_FILE_SIZE = 1024,
    DROPZONE_TIMEOUT = 5*60*1000)
   
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def RemoveBlackBorders(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)

def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def main():
    return render_template('index3.html')
       

@app.route("/upload",methods=["POST","GET"])
def upload():
    global image_path_screenshot
    if request.method == 'POST':
        img1= request.files.get('file')
        image_path_screenshot="images/screenshots/"+img1.filename
        #im = Image.open(img1.filename).convert("RGB")
        #im = RemoveBlackBorders(im)
        #width, height = im.size
        #print(width)
        
        img1.save(image_path_screenshot)
    return render_template('index3.html')

@app.route("/upload2",methods=["POST","GET"])
def upload2():
    global image_path_reference
    if request.method == 'POST':
        img2 = request.files.get('file')
        image_path_reference="images/reference image/"+img2.filename
        img2.save(image_path_reference)
    
    return render_template('index3.html')

@app.route("/",methods=["POST"])
def compare():
    if request.method == 'POST':
        image_names = [image_path_screenshot,image_path_reference]
        
        encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
        print(image_names)
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

        if score<0.89:
            prediction = "Not Matched - - "+ str(score)
        else:
            prediction = "Matched - - " + str(score)
        print(prediction)
        return render_template('index3.html',pred=prediction)
    else:
        prediction=''
        return render_template('index3.html')

    return render_template('index3.html')


if __name__ == "__main__":
    app.run(debug=True)
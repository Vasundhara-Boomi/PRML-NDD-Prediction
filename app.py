from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from transformers import pipeline
from pathlib import Path
import pymongo
import numpy as np

app = Flask(__name__)

# Load the image classification pipeline
pipe = pipeline("image-classification", "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")

# Connection string
connection_string = "mongodb+srv://sem6:ssn@cluster0.q0hpe8v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = pymongo.MongoClient(connection_string)

# Access a specific database
db = client['ndd_prediction']

# Access a specific collection within the database
collection = db['Credentials']

for document in collection.find():
    print(document)

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password=int(password)
        
        if username and password:
            # Query MongoDB for the username and password
            user = collection.find_one({"Username": username, "Password": password})
            
            if user:
                # If user exists, render the home page template
                print("sucess")
                return redirect(url_for('index'))
    
    # If user does not exist or credentials are incorrect, redirect back to the login page
    return render_template('login.html')

@app.route('/index')  # Define the route URL path
def index():
    # You can render the index.html template here
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        if file:
            # Save the image temporarily
            image_path = Path('temp.png')
            file.save(image_path)

            # Open and resize the image
            image = Image.open(image_path).resize((200, 200))

            # Make prediction using the model
            predictions = pipe(image)

            # Delete the temporary image file
            image_path.unlink()

            # Format the predictions
            formatted_predictions = []
            for prediction in predictions:
                label = prediction['label']
                score = prediction['score']
                formatted_prediction = f"{label} => 'score': {score}"
                formatted_predictions.append(formatted_prediction)

            # Apply conditional formatting for 'parkinson' label
            for i, prediction in enumerate(predictions):
                if prediction['label'] == 'parkinson' and prediction['score'] > 0.5:
                    formatted_predictions[i] = f"<span style='color:red'>{formatted_predictions[i]}</span>"

            return '<br>'.join(formatted_predictions)
    
    return 'Error'

#Allow files with IMGension png, jpg and jpeg
ALLOWED_IMG = set(['jpg' , 'jpeg' , 'png'])
ALLOWED_FILE = set(['csv'])

def allowed_img(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_IMG
           
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_FILE

# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


@app.route('/predict_2',methods=['GET','POST'])
def predict_2():
    if request.method == 'POST':
        file = request.files['file']
        acc = request.files['accelerometer']
        
        if (file and allowed_img(file.filename)) or (acc and allowed_file(acc.filename)): #Checking file format
            return render_template('result.html',prob="100%",user_image="static/img/image_1.jpg", img_name=file.filename, acc_name=acc.filename)
            
        else:
            return "Unable to read the file. Please check file IMGension"

if __name__ == '__main__':
    app.run(debug=True)

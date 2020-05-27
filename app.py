from flask import Flask, request, jsonify, make_response
from pycaret.classification import *
import pickle
import numpy as np
import pandas as pd

# Bug in Flask - have to add these two lines before importing flask_restplus
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

# Note that flask_restplus requires classes for all routes, not just functions
from flask_restplus import Api, Resource, fields

app = Flask(__name__)
api = Api(app = app, 
		  version = "1.0", 
		  title = "Project3 Flask App", 
		  description = "Predict results using a trained model")

# PyCaret automatically adds ".pkl"
model_file = 'models/et20200526_2010'

# Holdout data
holdoutData = 'data/holdout-2012.csv'

# Random generated data
baseFolder = "data/RandomSamples"

# Target column
target = 'zeroBalCode'

# List the features in the model
feature_cols = [
    'origChannel'
    , 'loanPurp'
    , 'bankNumber'
    , 'stateNumber'
    , 'mSA'
    , 'origIntRate'
    , 'origUPB'
    , 'origLTV'
    , 'numBorrowers'
    , 'origDebtIncRatio'
    , 'worstCreditScore'
]

generate_fields = api.model('Generate Files', {
    'howManyFiles': fields.Integer(
			required = True
			, description="How many files should we generate?"
			, help="Required field"
			, min=1
			, max=10
		)
	, 'howManyRows': fields.Integer(
			required = True
			, description="How many rows in each file?"
			, help="Required field"
			, min=10
			, max=100
		)
	, 'fileName': fields.String(
			required = True
			, description="What should the files start with?"
			, help="Required field"
		)
})

predict_fields = api.model('Predict Inputs', {
    'randomFile': fields.String(
			required = True
			, description="Of the generated files, which one do you choose?"
			, help="Required field"
		)
})

#########################################################    
# Pass in the answers and this method will create the 
#    test files from the holdout data
#########################################################
# @app.route('/predict_generate_files',methods=['POST'])
# @return_json
@api.route("/generate")
class GenFiles(Resource):
	def get(self):
		response = make_response()
		
		response = jsonify({ \
			"status": "Error - GET not implemented"
			, "info": "Using POST, pass howManyFiles, howManyRows, fileName" \
		})

		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")

		return response
	
#	@api.marshal_list_with(generate_fields)
	@api.doc(body=generate_fields)
#	@api.doc(parser=parser_generate)
	def put(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")

		try:
			formData = request.json
			howManyFiles = formData["howManyFiles"]
			howManyRows = formData["howManyRows"]
			fileName = formData["fileName"]
		except: 
			api.abort(500, "Form must pass howManyFiles, howManyRows, and fileName")

		try:
			howManyFiles = int(howManyFiles)
		except: 
			api.abort(500, "howManyFiles is not an integer")
		
		if howManyFiles < 1 or howManyFiles > 10:
			api.abort(500, "howManyFiles must be between 1 and 10")
			
		try:
			howManyRows = int(howManyRows)
		except: 
			api.abort(500, "howManyRows is not an integer")
		
		if howManyRows < 1 or howManyRows > 10:
			api.abort(500, "howManyRows must be between 1 and 10")
		
		if len(fileName) < 3 or len(fileName) > 10:
			api.abort(500, "fileName must be between 3 and 10 characters")

		try: 
			# Read in the holdout data into a Pandas DataFrame
			df = pd.read_csv(holdoutData)
		except: 
			api.abort(500, f"Unable to read holdoutData {holdoutData}")

		# Create the files withstarting with 01
		i = 1
		json_filenames = []

		while i <= howManyFiles:
			# Make sortable filenames (01, 02, 03 instead of 1, 2, 3)
			namingNumber = "01"

			if i < 10:
				namingNumber = "0" + str(i)
			else:
				namingNumber = str(i)
			
			the_file = f'{baseFolder}/{fileName}{namingNumber}.csv'

			try:
			# Step 1: Let's delete any previous runs' files first:
				os.remove(the_file)
			except:
				pass # How to do an empty except in Python

			# So we can pass what the names of the files created are back to front end
			file_dict = {}
			file_dict["fileName"] = the_file
			json_filenames.append(file_dict)

			# Export to csv
			df.sample(howManyRows).to_csv(the_file)

			# Get the next file or exit if processed last requested file
			i = i+1

		response = jsonify({"statusCode": 200, "Files": json_filenames})

		return response

#########################################################
# Pass in the name of the file selected and the model will
#    predict for you
#########################################################
# @app.route('/predict',methods=['POST'])
# @return_json
@api.route("/predict")
class PredictClass(Resource):
	def get(self):
		response = make_response()
		
		response = jsonify({ \
			"status": "Error - GET not implemented"
			, "info": "Using POST, pass randomFile" \
		})

		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")

		api.abort(404, "GET not implemented")

#	@api.marshal_list_with(predict_fields)
	@api.doc(body=predict_fields)
	#@api.doc(parser=parser_randomFile)
	# @api.param('randomFile', 'The name of your file')
	@api.doc(responses={404: 'randomFile not found'}, params={'randomFile': 'Name of the .csv file (MyFile01.csv)'})
	def put(self, randomFile):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		
		# args = parser.parse_args(strict=True)

		try:
			data_unseen = pd.read_csv(f'{baseFolder}/{randomFile}')
			# return jsonify({"randomFile": randomFile})
		except:			
			api.abort(404, f"File {randomFile} doesn't exist")

		# Load the model w PyCaret
		model = load_model(model_file)
		dfPredictions = predict_model(model, data=data_unseen)
		# prediction = int(prediction.Label[0])    
		# output = prediction.Label[0]

		# Remove the previous index columns
		dfPredictions.drop(['Unnamed: 0'], 1, inplace=True)
		dfPredictions.drop(['Unnamed: 0.1'], 1, inplace=True)

		response = jsonify({ \
			"statusCode": 200  \
			, "data": dfPredictions.to_json()  \
		})
		return response

api.add_resource(PredictClass, '/predict/<string:randomFile>')

if __name__ == '__main__':
    app.run(debug=True)
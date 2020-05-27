from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from pycaret.classification import *
import pickle
import numpy as np
import pandas as pd

flask_app = Flask(__name__)

# Load the model w PyCaret
# PyCaret automatically adds ".pkl"
model = load_model('models/et20200526_2010')

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

@app.route('/')
def home():	
	response = make_response()
	response.headers.add("Access-Control-Allow-Origin", "*")
	response.headers.add('Access-Control-Allow-Headers', "*")
	response.headers.add('Access-Control-Allow-Methods', "*")

	routes = {}
	route_dict = {}
	route_dict["01-GenerateFiles"] = "/predict_generate_files" 
	route_dict["02-Predict"] = "/predict" 
	routes.append(route_dict)

	response = jsonify({ \
		"statusCode": 200 \
		, "status": "Site is up!" \
		, "asof": str(data) \
		, "routes": routes \
	})
	
	return response

#########################################################    
# Pass in the answers and this method will create the 
#    test files from the holdout data
#########################################################
@app.route('/predict_generate_files',methods=['POST'])
@return_json
def predict(files, rows_per_file, fileName):
	response = make_response()
	response.headers.add("Access-Control-Allow-Origin", "*")
	response.headers.add('Access-Control-Allow-Headers', "*")
	response.headers.add('Access-Control-Allow-Methods', "*")
	
    # Read in the holdout data into a Pandas DataFrame
	df = pd.read_csv(holdoutData)

	# Create the files withstarting with 01
	i = 1
	json_filenames = []

	while i <= files:		
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
		df.sample(rows_per_file).to_csv(the_file)
		i = i+1

    response = jsonify({"statusCode": 200, "Files": json_filenames})
        
    return response

#########################################################
# Pass in the name of the file selected and the model will
#    predict for you
#########################################################
@app.route('/predict',methods=['POST'])
@return_json
def predict(chosen_file):
	response = make_response()
	response.headers.add("Access-Control-Allow-Origin", "*")
	response.headers.add('Access-Control-Allow-Headers', "*")
	response.headers.add('Access-Control-Allow-Methods', "*")

	try:
    	data_unseen = pd.read_csv(f'{baseFolder}/{chosen_file}')

	except:
		response = jsonify({ \
			"statusCode": 404 \
			, "status": f"Unable to find file '{chosen_file}'" \
		})
		return response

    dfPredictions = predict_model(model, data=data_unseen, round = 4)
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

if __name__ == '__main__':
    app.run(debug=True)









=========== ML App app.py

app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "ML React App", 
		  description = "Predict results using a trained model")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'textField1': fields.String(required = True, 
				  							   description="Text Field 1", 
    					  				 	   help="Text Field 1 cannot be blank"),
				  'textField2': fields.String(required = True, 
				  							   description="Text Field 2", 
    					  				 	   help="Text Field 2 cannot be blank"),
				  'select1': fields.Integer(required = True, 
				  							description="Select 1", 
    					  				 	help="Select 1 cannot be blank"),
				  'select2': fields.Integer(required = True, 
				  							description="Select 2", 
    					  				 	help="Select 2 cannot be blank"),
				  'select3': fields.Integer(required = True, 
				  							description="Select 3", 
    					  				 	help="Select 3 cannot be blank")})

# classifier = joblib.load('classifier.joblib')

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try: 
			formData = request.json
			data = [val for val in formData.values()]
			# prediction = classifier.predict(data)
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Prediction: " + str(data)
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
# run with ". curl_tests.sh"

#####################################################
# /predict
#####################################################
# GET
curl "http://127.0.0.1:5000/predict"

# PUT - passing right parameter
curl "http://127.0.0.1:5000/predict/testing05.csv" -X PUT

#####################################################
# /generate
#####################################################
# /generate - GET
curl "http://127.0.0.1:5000/generate"

# /generate - PUT - passing no parameters
# howManyFiles, howManyRows, fileName
curl "http://127.0.0.1:5000/generate" -d '{"howManyFiles":"2", "howManyRows":"10", "fileName": "Scott"}' -H "Content-Type: application/json" -X PUT

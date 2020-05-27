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
curl "http://127.0.0.1:5000/generate" -d"howManyFiles=10" -d"howManyRows=10" -d"fileName=Scott.txt" -X PUT
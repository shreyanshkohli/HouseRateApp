from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_home_price', methods=['GET','POST'])
def predict_home_price():
    # Try to handle form data or JSON data
    try:
        if request.content_type == 'application/json':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            total_sqft = float(data.get('total_sqft', 0))
            location = data.get('location', '')
            bedrooms = int(data.get('bedrooms', 0))
            bath = int(data.get('bath', 0))
        else:
            total_sqft = float(request.form.get('total_sqft', 0))
            location = request.form.get('location', '')
            bedrooms = int(request.form.get('bedrooms', 0))
            bath = int(request.form.get('bath', 0))
    
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {e}'}), 400
    except TypeError as e:
        return jsonify({'error': f'Invalid type: {e}'}), 400

    try:
        # Call the utility function to get the estimated price
        prediction = util.pricePredict(location, total_sqft, bedrooms, bath)
        response = jsonify({'estimated_price': prediction})
    except Exception as e:
        return jsonify({'error': f'Internal server error: {e}'}), 500

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True, port=5001, host='0.0.0.0')

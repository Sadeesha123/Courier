import cv2 as cv
import numpy as np 
import tensorflow as tf
from flask_cors import CORS
import folium, requests, time
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

class_dict_defect = {
                    'defective': 0, 
                    'qualified': 1
                    }

class_dict_defect_rev = {
                        0: 'defective',
                        1: 'qualified'
                        }

model_defects = tf.keras.models.load_model('models/courier-defect-detector.h5')


def inference_model(
                    img_path,
                    target_size = (224, 224)
                    ):
    img = cv.imread(img_path)
    img = cv.resize(img, target_size)
    img = tf.keras.applications.xception.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model_defects.predict(img)
    pred = pred.squeeze() > 0.5
    pred = pred.squeeze()
    return class_dict_defect_rev[int(pred)]

def get_polyline(
                start, end, 
                api_key = 'G8pmgqBjIXhCKZ5ABCaq0HrpqWwBiMg8', 
                max_attempts=3, 
                delay=1
                ):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start}:{end}/json?routeType=shortest&traffic=true&key={api_key}&maxAlternatives=4"
    attempts = 0

    while attempts < max_attempts:
        response = requests.get(url)
        data = response.json()
        
        polylines, lengthsInMeters, travelTimesInSeconds = [], [], []
        if len(data['routes']) > 0:
            for route in data['routes']:
                p_line = route['legs'][0]['points']
                p_line = [(p['latitude'], p['longitude']) for p in p_line]
                polylines.append(p_line)
                lengthsInMeters.append(route['summary']['lengthInMeters'])
                travelTimesInSeconds.append(route['summary']['travelTimeInSeconds'])

            return polylines, lengthsInMeters, travelTimesInSeconds

        attempts += 1
        time.sleep(delay)

    print(f"Skipped delivery row: {start} -> {end} (Polyline not available)")
    return None, None, None

def visualize_route(pickup, delivery):
    try:
        pickup = eval(pickup)
    except:
        pass 

    try:
        delivery = eval(delivery)
    except:
        pass

    pickup_str = f"{pickup[0]},{pickup[1]}"
    delivery_str = f"{delivery[0]},{delivery[1]}"
    polylines, _, travelTimesInSeconds = get_polyline(pickup_str, delivery_str)
    best_line_idx = np.argmin(travelTimesInSeconds)

    map = folium.Map(location=[np.mean([pickup[0], delivery[0]]), np.mean([pickup[1], delivery[1]])], zoom_start=12)
    folium.Marker(pickup, icon=folium.Icon(color='blue', icon='truck', prefix='fa')).add_to(map)
    folium.Marker(delivery, icon=folium.Icon(color='blue', icon='truck', prefix='fa')).add_to(map)

    for i in range(len(polylines)):
        polyline = polylines[i]
        if i == best_line_idx:
            pass
        else:
            for p in polyline:
                folium.CircleMarker(p, radius=2, color='green').add_to(map)

    best_polyline = polylines[best_line_idx]
    for p in best_polyline:
        folium.CircleMarker(p, radius=2, color='red').add_to(map)

    map.save('maps/map-inference.html')
    return 'maps/map-inference.html'

@app.route('/defects', methods=['POST'])
def defects():
    if request.method == 'POST':
        data = request.files
        image = data['image']
        
        filename = f"uploads/{image.filename}"
        image.save(filename)

        result = inference_model(filename)
        return jsonify({
                        "status": "success",
                        "defectiveness": result,
                        }), 200
    
    return jsonify({
                    "status": "unsuccess",
                    "defectiveness": None,
                    }), 400

@app.route('/maps', methods=['POST'])
def maps():
    if request.method == 'POST':
        data = request.form
        pickup = data['pickup']
        delivery = data['delivery']

        result = visualize_route(pickup, delivery)
        return jsonify({
                        "status": "success",
                        "map": result,
                        }), 200
    
    return jsonify({
                    "status": "unsuccess",
                    "map": None,
                    }), 400

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

reference_descriptors = None
reference_keypoints = None
reference_image = None

sift = cv2.SIFT_create()

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    global reference_descriptors, reference_keypoints, reference_image

    data = request.get_json()
    if data is None:
        return jsonify({
            'error': 'Invalid or missing JSON in upload_reference',
            'raw_request': request.data.decode('utf-8', errors='ignore'),
            'headers': dict(request.headers)
        }), 400

    base64_img = data.get('image') or data.get('imageBase64')
    if not base64_img:
        return jsonify({'error': 'Missing image data'}), 400

    image = base64_to_image(base64_img)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return jsonify({'error': 'No features found in reference image'}), 400

    reference_image = image
    reference_descriptors = descriptors
    reference_keypoints = keypoints
    return jsonify({'status': 'Reference descriptors stored'})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({
                'status': 'fail',
                'reason': 'request.get_json() returned None',
                'content_type': request.headers.get('Content-Type'),
                'raw_body': request.data.decode('utf-8', errors='replace'),
            }), 400

        return jsonify({
            'status': 'ok',
            'keys': list(data.keys()),
            'example_value_start': str(data)[:200],
            'content_type': request.headers.get('Content-Type'),
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'exception',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

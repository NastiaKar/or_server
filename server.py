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
    global reference_descriptors, reference_keypoints, reference_image

    if reference_descriptors is None:
        return jsonify({'error': 'No reference uploaded'}), 400

    try:
        data = request.get_json()
        if data is None:
            return jsonify({
                'error': 'Invalid or missing JSON',
                'raw_body': request.data.decode('utf-8', errors='ignore'),
            }), 400

        base64_img = data.get('image') or data.get('imageBase64')
        if not base64_img:
            return jsonify({'error': 'Missing image data'}), 400

        image = base64_to_image(base64_img)
        kp2, des2 = sift.detectAndCompute(image, None)
        if des2 is None:
            return jsonify({'match': False, 'reason': 'No features in camera image'})

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(reference_descriptors, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return jsonify({
                'match': False,
                'reason': 'Not enough good matches',
                'good_matches': len(good_matches)
            })

        src_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        is_match = inliers >= 10 and (inliers / len(good_matches)) >= 0.3

        box_points = None
        if is_match and mask is not None:
            matched_pts = np.float32([kp2[m.trainIdx].pt for i, m in enumerate(good_matches) if mask[i]])
            if len(matched_pts) >= 4:
                rect = cv2.minAreaRect(matched_pts)
                box = cv2.boxPoints(rect)
                box_points = box.tolist()

        return jsonify({
            'match': is_match,
            'good_matches': len(good_matches),
            'total_matches': len(matches),
            'inliers': inliers,
            'box': box_points
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import imutils
import easyocr
import uuid  # Import modul uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Fungsi untuk memastikan direktori sudah ada
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # Pengecekan ekstensi file yang diizinkan
    if not allowed_file(file.filename):
        return render_template('result.html', image_path=None, message="Format file tidak diizinkan. Hanya diperbolehkan file dengan ekstensi 'jpg', 'jpeg', atau 'png'.")

    filename = secure_filename(file.filename)

    # Menghasilkan nama file unik dengan UUID
    unique_filename = str(uuid.uuid4()) + "_" + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # Logika OpenCV di sini
    img = cv2.imread(filepath)

    if img is None:
        return render_template('result.html', image_path=None, message="File tidak dapat dibaca atau bukan file gambar.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None

    # Pengecekan apakah plat nomor terdeteksi
    if len(contours) > 0:
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 2, y1:y2 + 2]

            reader = easyocr.Reader(['id'])
            result = reader.readtext(cropped_image)

            # Pemeriksaan agar tidak terjadi IndexError
            if result and len(result) > 0:
                text = result[0][-2]
            else:
                text = "Tidak dapat membaca teks pada gambar."

            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                              color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

            # Memastikan direktori penyimpanan hasil sudah ada
            ensure_dir(app.config['RESULT_FOLDER'])

            # Menyimpan hasil gambar dengan nama unik
            unique_result_filename = str(uuid.uuid4()) + "_result.jpg"
            result_path = os.path.join(app.config['RESULT_FOLDER'], unique_result_filename)
            cv2.imwrite(result_path, res)

            return render_template('result.html', image_path=result_path, message=None)
        else:
            # Plat nomor tidak terdeteksi dengan benar
            return render_template('result.html', image_path=None, message="Plat nomor tidak terdeteksi dengan benar oleh logika OpenCV.")
    else:
        # Plat nomor tidak terdeteksi
        return render_template('result.html', image_path=None, message="Plat nomor tidak terdeteksi atau tidak sesuai dengan ketentuan.")

@app.route('/result_images')
def result_images():
    result_images_path = app.config['RESULT_FOLDER']
    return render_template('result_images.html', result_images=os.listdir(result_images_path))

@app.route('/result_images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/delete_image/<filename>')
def delete_image(filename):
    result_images_path = app.config['RESULT_FOLDER']
    file_path = os.path.join(result_images_path, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return redirect(url_for('result_images'))

    return render_template('result_images.html', result_images=os.listdir(result_images_path), message="File not found")

if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
import cv2
import numpy as np
import imutils
import easyocr
import os

# Fungsi untuk memastikan direktori sudah ada
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Fungsi deteksi plat nomor
def detect_license_plate(img):
    # Logika OpenCV di sini
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
                text = " ".join([res[-2] for res in result])  # Menggabungkan semua teks menjadi satu baris
            else:
                text = "Tidak dapat membaca teks pada gambar."

            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                              color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

            return res, text
        else:
            return None, "Plat nomor tidak terdeteksi dengan benar oleh logika OpenCV."
    else:
        return None, "Plat nomor tidak terdeteksi atau tidak sesuai dengan ketentuan."

def delete_result_image(filename):
    result_images_path = 'static/results'
    file_path = os.path.join(result_images_path, filename)

    try:
        os.unlink(file_path)
        return True
    except OSError:
        return False
    
def main():
    st.title('ANPR App with Streamlit')

    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if img is not None:
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            st.write("Classifying...")

            result, text = detect_license_plate(img)

            if result is not None:
                st.image(result, caption=f"Result: {text}", use_column_width=True)
                st.success(f"ANPR Result: {text}")
            else:
                st.error(text)

    if st.button("Show Result Images"):
        result_images_path = 'static/results'
        result_images = os.listdir(result_images_path)

        if result_images:
            for result_image in result_images:
                st.image(f"{result_images_path}/{result_image}", caption=result_image, use_column_width=True)

                # Tambahkan tombol hapus untuk setiap gambar hasil
                if st.button(f"Delete {result_image}"):
                    if delete_result_image(result_image):
                        st.success(f"File {result_image} berhasil dihapus.")
                    else:
                        st.warning(f"File {result_image} tidak ditemukan atau tidak dapat dihapus.")
        else:
            st.warning("No result images available.")

if __name__ == '__main__':
    main()

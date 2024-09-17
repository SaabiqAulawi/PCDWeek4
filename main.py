from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import cv2
import os
import numpy as np
import base64

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse('index.html')

def process_image(image):
    # Menambahkan noise ke gambar
    noisy_img = add_noise(image, 0.05)

    # Menghilangkan noise dari gambar
    denoised_img = remove_noise(noisy_img)

    # Menajamkan gambar
    sharpened_img = sharpen_image(denoised_img)

    return noisy_img, denoised_img, sharpened_img

def add_noise(img, prob):
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output

def remove_noise(img):
    return cv2.medianBlur(img, 5)

def sharpen_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened

@app.post("/capture-images/")
async def capture_images(request: Request):
    data = await request.json()
    person_name = data['person_name']
    image_data = data['image_data']

    save_path = os.path.join('dataset', person_name)
    processed_path = os.path.join('processed_dataset', person_name)

    # Membuat folder 'dataset' dan 'processed_dataset' jika belum ada
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    if not os.path.exists('processed_dataset'):
        os.makedirs('processed_dataset')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    # Dekode gambar dari base64
    image_data = image_data.split(",")[1]  # Remove data:image/jpeg;base64,
    img_bytes = base64.b64decode(image_data)
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Simpan gambar asli ke folder 'dataset'
    img_name = os.path.join(save_path, f"{person_name}_{len(os.listdir(save_path))}.jpg")
    cv2.imwrite(img_name, img)

    # Lakukan pengolahan citra
    noisy_img, denoised_img, sharpened_img = process_image(img)

    # Simpan hasil pengolahan ke folder 'processed_dataset'
    noisy_img_name = os.path.join(processed_path, f"noisy_{person_name}_{len(os.listdir(save_path))}.jpg")
    denoised_img_name = os.path.join(processed_path, f"denoised_{person_name}_{len(os.listdir(save_path))}.jpg")
    sharpened_img_name = os.path.join(processed_path, f"sharpened_{person_name}_{len(os.listdir(save_path))}.jpg")

    cv2.imwrite(noisy_img_name, noisy_img)
    cv2.imwrite(denoised_img_name, denoised_img)
    cv2.imwrite(sharpened_img_name, sharpened_img)

    # Encode gambar hasil pengolahan
    _, noisy_img_encoded = cv2.imencode('.jpg', noisy_img)
    _, denoised_img_encoded = cv2.imencode('.jpg', denoised_img)
    _, sharpened_img_encoded = cv2.imencode('.jpg', sharpened_img)

    # Mengubah hasil encode menjadi base64
    noisy_img_base64 = base64.b64encode(noisy_img_encoded).decode('utf-8')
    denoised_img_base64 = base64.b64encode(denoised_img_encoded).decode('utf-8')
    sharpened_img_base64 = base64.b64encode(sharpened_img_encoded).decode('utf-8')

    return JSONResponse(content={
        "detail": "Gambar berhasil disimpan dan diolah.",
        "noisy_image": f"data:image/jpeg;base64,{noisy_img_base64}",
        "denoised_image": f"data:image/jpeg;base64,{denoised_img_base64}",
        "sharpened_image": f"data:image/jpeg;base64,{sharpened_img_base64}"
    })

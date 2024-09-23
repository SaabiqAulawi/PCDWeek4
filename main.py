from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import numpy as np
import base64

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse('index2.html')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def add_salt_and_pepper(image, prob):
    output = np.copy(image)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
    return output

def count_noise(original, noisy):
    return np.sum(np.any(original != noisy, axis=-1))

def change_noise_color(image, original, salt_color, pepper_color):
    output = np.copy(image)
    salt_mask = np.all(image == 255, axis=-1) & np.any(original != 255, axis=-1)
    pepper_mask = np.all(image == 0, axis=-1) & np.any(original != 0, axis=-1)
    output[salt_mask] = salt_color
    output[pepper_mask] = pepper_color
    return output

def remove_noise(img):
    return cv2.medianBlur(img, 5)

@app.post("/process-image/")
async def process_image(request: Request):
    data = await request.json()
    image_data = data['image_data']
    noise_prob = float(data['noise_prob'])
    salt_color = np.array(list(map(int, data['salt_color'].split(','))), dtype=np.uint8)
    pepper_color = np.array(list(map(int, data['pepper_color'].split(','))), dtype=np.uint8)
    person_name = data['person_name']

    ensure_dir('dataset')
    ensure_dir('processed_dataset')
    ensure_dir(os.path.join('dataset', person_name))
    ensure_dir(os.path.join('processed_dataset', person_name))

    image_data = image_data.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    img_np = np.frombuffer(img_bytes, np.uint8)
    original_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    original_filename = f"{person_name}_{len(os.listdir(os.path.join('dataset', person_name)))}.jpg"
    cv2.imwrite(os.path.join('dataset', person_name, original_filename), original_img)

    # Add salt and pepper noise
    noisy_img = add_salt_and_pepper(original_img, noise_prob)

    # Count noise pixels
    noise_count = count_noise(original_img, noisy_img)

    # Change noise color
    colored_noise_img = change_noise_color(noisy_img, original_img, salt_color, pepper_color)

    # Remove noise
    denoised_img = remove_noise(colored_noise_img)

    cv2.imwrite(os.path.join('processed_dataset', person_name, f"noisy_{original_filename}"), noisy_img)
    cv2.imwrite(os.path.join('processed_dataset', person_name, f"colored_{original_filename}"), colored_noise_img)
    cv2.imwrite(os.path.join('processed_dataset', person_name, f"denoised_{original_filename}"), denoised_img)

    # Encode processed images for response
    _, noisy_img_encoded = cv2.imencode('.jpg', noisy_img)
    _, colored_noise_img_encoded = cv2.imencode('.jpg', colored_noise_img)
    _, denoised_img_encoded = cv2.imencode('.jpg', denoised_img)

    # Convert to base64
    noisy_img_base64 = base64.b64encode(noisy_img_encoded).decode('utf-8')
    colored_noise_img_base64 = base64.b64encode(colored_noise_img_encoded).decode('utf-8')
    denoised_img_base64 = base64.b64encode(denoised_img_encoded).decode('utf-8')

    return JSONResponse(content={
        "noisy_image": f"data:image/jpeg;base64,{noisy_img_base64}",
        "colored_noise_image": f"data:image/jpeg;base64,{colored_noise_img_base64}",
        "denoised_image": f"data:image/jpeg;base64,{denoised_img_base64}",
        "noise_count": int(noise_count),
        "original_filename": original_filename
    })

@app.post("/capture-images/")
async def capture_images(request: Request):
    data = await request.json()
    person_name = data['person_name']
    image_data = data['image_data']

    ensure_dir('dataset')
    ensure_dir(os.path.join('dataset', person_name))

    # Decode image from base64
    image_data = image_data.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    filename = f"{person_name}_{len(os.listdir(os.path.join('dataset', person_name)))}.jpg"
    cv2.imwrite(os.path.join('dataset', person_name, filename), img)

    return JSONResponse(content={
        "detail": f"Image saved as {filename}",
        "filename": filename
    })
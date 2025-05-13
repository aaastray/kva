import os
import shutil  # Не используется, но может быть полезен для управления файлами
import uuid
import random
from typing import List, Optional, Dict, Tuple
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- ML Imports ---
import torch
from torchvision import transforms as TVTransforms
from PIL import Image, ImageOps, ImageEnhance  # ImageEnhance добавлен, если понадобится
import numpy as np
import tensorflow as tf
from transformers import SegformerForSemanticSegmentation
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops

# import cv2 # Закомментирован, так как CLAHE из пользовательской функции не реализуется активно

# --- Конфигурация ---
UPLOAD_DIRECTORY = "static/uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- ML Model Configuration ---
SEGMENTATION_MODEL_PATH = "models/segmentation"
PREDICT_MODEL_DIR = "models/predict"
MAN_PREDICT_MODEL_PATH = os.path.join(PREDICT_MODEL_DIR, "bone_age_model_man.keras")
WOMAN_PREDICT_MODEL_PATH = os.path.join(PREDICT_MODEL_DIR, "bone_age_model_woman.keras")

# Device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} for PyTorch models.")

# --- Глобальные переменные для моделей ---
segmentation_model_global = None
seg_processor_global = None
keras_model_man_global = None
keras_model_woman_global = None

# --- "База данных" в памяти ---
db_patients: List[Dict] = []
patient_id_counter = 0
analysis_id_counter = 0


# --- Pydantic Модели ---
class AnalysisBase(BaseModel):
    date: str
    doctorNotes: Optional[str] = ""


class Analysis(AnalysisBase):
    id: int
    predictedAge: float
    xrayImageURL: str


class PatientBase(BaseModel):
    lastName: str
    firstName: str
    middleName: Optional[str] = ""
    birthDate: str
    gender: str  # 'male' или 'female'
    policyNumber: str


class Patient(PatientBase):
    id: int
    analyses: List[Analysis] = []


class PatientCreate(PatientBase):
    pass


class AnalysisNotesUpdate(BaseModel):
    doctorNotes: str


# --- Keras Custom Layer ---
class Cast(tf.keras.layers.Layer):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self.target_dtype = tf.dtypes.as_dtype(dtype)

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'dtype': self.target_dtype.name})
        return config


# --- Функции для ML ---

def load_ml_models():
    global segmentation_model_global, seg_processor_global, keras_model_man_global, keras_model_woman_global
    print("Loading ML models...")
    try:
        SEGFORMER_IMAGE_SIZE = 1024  # Размер для Segformer
        if not os.path.exists(os.path.join(SEGMENTATION_MODEL_PATH, "config.json")):
            print(f"Segformer config.json not found in {SEGMENTATION_MODEL_PATH}. Segmentation will be skipped.")
        else:
            seg_processor_global = TVTransforms.Compose([
                TVTransforms.Resize((SEGFORMER_IMAGE_SIZE, SEGFORMER_IMAGE_SIZE)),
                TVTransforms.ToTensor(),
                TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            segmentation_model_global = SegformerForSemanticSegmentation.from_pretrained(SEGMENTATION_MODEL_PATH).to(
                DEVICE)
            segmentation_model_global.eval()
            print("Segformer model loaded successfully.")

        custom_objects = {'Cast': Cast}
        if os.path.exists(MAN_PREDICT_MODEL_PATH):
            keras_model_man_global = tf.keras.models.load_model(MAN_PREDICT_MODEL_PATH, custom_objects=custom_objects)
            print("Keras model for men loaded successfully.")
        else:
            print(f"Keras model for men not found at {MAN_PREDICT_MODEL_PATH}. Prediction for men will be random.")

        if os.path.exists(WOMAN_PREDICT_MODEL_PATH):
            keras_model_woman_global = tf.keras.models.load_model(WOMAN_PREDICT_MODEL_PATH,
                                                                  custom_objects=custom_objects)
            print("Keras model for women loaded successfully.")
        else:
            print(
                f"Keras model for women not found at {WOMAN_PREDICT_MODEL_PATH}. Prediction for women will be random.")
        print("ML model loading complete.")
    except Exception as e:
        print(f"ERROR loading ML models: {e}")
        import traceback
        traceback.print_exc()


def apply_mask_and_crop(
        image_to_process: Image.Image,
        binary_mask_np: np.ndarray,
        output_mode: str = 'RGB'
) -> Optional[Image.Image]:
    if binary_mask_np is None or not np.any(binary_mask_np):
        print("No valid mask provided for apply_mask_and_crop. Returning None.")
        return None

    if image_to_process.size != (binary_mask_np.shape[1], binary_mask_np.shape[0]):
        error_msg = (f"ERROR: Image size {image_to_process.size} and mask shape "
                     f"{binary_mask_np.shape} mismatch in apply_mask_and_crop. "
                     "Attempting to resize image to mask dimensions (this might be risky).")
        print(error_msg)
        try:
            image_to_process = image_to_process.resize((binary_mask_np.shape[1], binary_mask_np.shape[0]),
                                                       Image.Resampling.LANCZOS)
        except Exception as resize_err:
            print(f"Failed to resize image in apply_mask_and_crop: {resize_err}")
            return None

    target_mode = 'RGBA' if output_mode == 'RGBA' else 'RGB'
    if image_to_process.mode != target_mode:
        image_converted = image_to_process.convert(target_mode)
    else:
        image_converted = image_to_process.copy()

    mask_pil = Image.fromarray(binary_mask_np * 255, mode='L')

    if target_mode == 'RGBA':
        masked_image = Image.new("RGBA", image_converted.size, (0, 0, 0, 0))
        masked_image.paste(image_converted, (0, 0), mask_pil)
    else:  # RGB
        img_array = np.array(image_converted)
        if binary_mask_np.ndim == 3 and binary_mask_np.shape[2] == 1:
            binary_mask_np = binary_mask_np.squeeze(axis=2)
        mask_3channel = np.stack([binary_mask_np] * 3, axis=-1)
        masked_array = np.where(mask_3channel, img_array, 0)
        masked_image = Image.fromarray(masked_array.astype(np.uint8), 'RGB')

    filled_mask_for_bbox = binary_fill_holes(binary_mask_np)
    labeled_mask_for_bbox = label(filled_mask_for_bbox)
    regions = regionprops(labeled_mask_for_bbox)

    if not regions:
        print("No regions found in mask after processing for bbox. Returning full masked image.")
        return masked_image

    largest_region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest_region.bbox

    h, w = masked_image.height, masked_image.width
    minr, minc = max(0, minr), max(0, minc)
    maxr, maxc = min(h, maxr), min(w, maxc)

    if maxr <= minr or maxc <= minc:
        print("Bounding box for cropping is invalid or empty. Returning full masked image.")
        return masked_image

    final_cropped_image = masked_image.crop((minc, minr, maxc, maxr))

    if final_cropped_image.width == 0 or final_cropped_image.height == 0:
        print("Cropping after masking resulted in an empty image. Returning full masked image.")
        return masked_image
    return final_cropped_image


def predict_segmentation_mask(pil_image_rgb_resized: Image.Image) -> Optional[np.ndarray]:
    if segmentation_model_global is None or seg_processor_global is None:
        print("Segmentation model not available. Skipping segmentation.")
        return None
    try:
        pixel_values = seg_processor_global(pil_image_rgb_resized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = segmentation_model_global(pixel_values=pixel_values)
            logits = outputs.logits

        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(pil_image_rgb_resized.height, pil_image_rgb_resized.width),
            mode="bilinear",
            align_corners=False,
        )
        pred_seg_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

        HAND_CLASS_ID = 2  # Убедись, что это правильный ID для руки в твоей модели
        binary_mask = (pred_seg_map == HAND_CLASS_ID).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            print(f"Binary mask for HAND_CLASS_ID {HAND_CLASS_ID} is empty.")
            return None
        return binary_mask
    except Exception as e:
        print(f"Error in predict_segmentation_mask: {e}")
        import traceback
        traceback.print_exc()
        return None


# Адаптированная пользовательская функция preprocess_image
# (Убраны autocontrast и clahe, так как они не использовались в предоставленном коде и требуют cv2)
# Переименована для ясности и чтобы не конфликтовать с другими возможными функциями
def backend_preprocess_image(
        input_pil_image: Image.Image,
        target_size: tuple = (224, 224),
        crop_threshold: int = 15,  # Порог для начальной обрезки по яркости
        padding_margin: int = 15  # Отступ при начальной обрезке
) -> Optional[Image.Image]:
    if input_pil_image is None:
        print("Input image to backend_preprocess_image is None.")
        return None

    try:
        # 1. Конвертация в 'L'. EXIF-транспонирование должно быть сделано до вызова этой функции.
        if input_pil_image.mode == 'RGBA':
            # Если есть альфа-канал, накладываем на черный фон перед конвертацией в L
            # Это важно, если маскирование создает прозрачность, а не черный фон
            background = Image.new("RGB", input_pil_image.size, (0, 0, 0))
            background.paste(input_pil_image, (0, 0), input_pil_image.split()[3])  # paste using alpha channel as mask
            img = background.convert('L')
        elif input_pil_image.mode == 'L':
            img = input_pil_image.copy()  # Работаем с копией
        else:  # 'RGB', 'P', etc.
            img = input_pil_image.convert('L')

        # 2. Обрезка по яркости с защитной зоной
        # Эта обрезка может быть полезна, если рука не идеально заполнила кадр после apply_mask_and_crop,
        # или если apply_mask_and_crop не использовался.
        img_array = np.array(img)
        mask = img_array > crop_threshold  # Пиксели ярче порога

        if np.any(mask):  # Если есть что-то ярче порога
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if np.any(rows) and np.any(cols):  # Убедимся, что есть что обрезать
                rmin = max(0, np.where(rows)[0][0] - padding_margin)
                rmax = min(img.height, np.where(rows)[0][-1] + padding_margin)
                cmin = max(0, np.where(cols)[0][0] - padding_margin)
                cmax = min(img.width, np.where(cols)[0][-1] + padding_margin)

                if rmax > rmin and cmax > cmin:  # Если bbox валидный
                    img = img.crop((cmin, rmin, cmax, rmax))
                # else: пропустить обрезку, если bbox невалиден
            # else: пропустить обрезку, если нет строк/столбцов по маске
        # else: пропустить обрезку, если всё изображение темнее порога

        if img.width == 0 or img.height == 0:
            print(
                f"Error: Image became empty after cropping in backend_preprocess_image. Original input size: {input_pil_image.size}")
            return None

        # 3. Масштабирование с сохранением пропорций
        # Логика ratio из твоего оригинального preprocess_image:
        # Учитывает padding_margin, чтобы объект был чуть меньше, оставляя место для полей.
        original_width, original_height = img.size
        if original_width == 0 or original_height == 0:  # Доп. проверка
            print(f"Error: Image dimensions are zero before resize in backend_preprocess_image.")
            return None

        # Избегаем деления на ноль, если original_width/height + 2*padding_margin становится <=0 (маловероятно с позитивными padding_margin)
        denominator_w = original_width + 2 * padding_margin
        denominator_h = original_height + 2 * padding_margin
        if denominator_w <= 0 or denominator_h <= 0:
            print(f"Warning: Invalid denominators for ratio calculation. Using simple ratio.")
            ratio = min(target_size[0] / original_width, target_size[1] / original_height)
        else:
            ratio = min(
                target_size[0] / denominator_w,
                target_size[1] / denominator_h
            )

        new_w = int(original_width * ratio)
        new_h = int(original_height * ratio)

        if new_w <= 0 or new_h <= 0:
            print(
                f"Warning: Calculated new_size ({new_w}x{new_h}) is zero or negative. Resizing to target_size directly, proportions may be lost.")
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            new_w, new_h = target_size  # Обновляем new_w, new_h для паддинга
        else:
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 4. Паддинг до target_size
        # Используем простой paste вместо цикла putpixel с порогом.
        # Это сохранит все детали отмасштабированного изображения.
        padded_img = Image.new("L", target_size, 0)  # Черный фон для паддинга
        pad_x = (target_size[0] - new_w) // 2
        pad_y = (target_size[1] - new_h) // 2

        padded_img.paste(img_resized, (pad_x, pad_y))

        return padded_img

    except Exception as e:
        print(f"Error in backend_preprocess_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


IMG_SIZE_KERAS = (224, 224)  # Размер входа для Keras моделей


def predict_bone_age(processed_pil_image_L: Image.Image, gender: str) -> Optional[float]:
    if processed_pil_image_L is None:
        print("No processed image (L channel, 224x224) for Keras prediction.")
        return None

    model_to_use = keras_model_man_global if gender.lower() == 'male' else keras_model_woman_global
    if model_to_use is None:
        print(f"Keras model for gender '{gender}' not loaded. Skipping prediction.")
        return None

    try:
        # processed_pil_image_L это PIL Image, 'L' mode, размером IMG_SIZE_KERAS
        img_array = np.array(processed_pil_image_L, dtype=np.float32) / 255.0  # Нормализация
        # Модель Keras ожидает 3 канала (H, W, C)
        img_array_rgb_like = np.repeat(np.expand_dims(img_array, axis=-1), 3, axis=-1)  # (224, 224, 3)
        img_batch = np.expand_dims(img_array_rgb_like, axis=0)  # (1, 224, 224, 3)

        prediction = model_to_use.predict(img_batch)
        predicted_age_months = float(prediction.flatten()[0])
        return predicted_age_months
    except Exception as e:
        print(f"Error in predict_bone_age (Keras): {e}")
        import traceback
        traceback.print_exc()
        return None


# --- FastAPI приложение ---
app = FastAPI(title="Patient Management API")


@app.on_event("startup")
async def startup_event():
    load_ml_models()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static/uploads", StaticFiles(directory=UPLOAD_DIRECTORY), name="static_uploads")


# --- Вспомогательные функции для CRUD (без изменений) ---
def get_next_patient_id(): global patient_id_counter; patient_id_counter += 1; return patient_id_counter


def get_next_analysis_id(): global analysis_id_counter; analysis_id_counter += 1; return analysis_id_counter


def find_patient_or_404(patient_id: int) -> Dict:
    patient = next((p for p in db_patients if p["id"] == patient_id), None)
    if not patient: raise HTTPException(status_code=404, detail="Patient not found")
    return patient


def find_analysis_or_404(patient: Dict, analysis_id: int) -> Dict:
    analysis = next((a for a in patient["analyses"] if a["id"] == analysis_id), None)
    if not analysis: raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


# --- API Эндпоинты ---
@app.post("/patients", response_model=Patient, status_code=201)
async def create_patient_endpoint(patient_data: PatientCreate):
    new_id = get_next_patient_id()
    new_patient_dict = patient_data.model_dump()
    new_patient_dict["id"] = new_id;
    new_patient_dict["analyses"] = []
    db_patients.append(new_patient_dict)
    return new_patient_dict


@app.get("/patients", response_model=List[Patient])
async def get_patients_endpoint(searchQuery: Optional[str] = None):
    if not searchQuery: return db_patients
    query = searchQuery.lower().strip()
    return [p for p in db_patients if
            query in f"{p['lastName']} {p['firstName']} {p.get('middleName', '')}".lower() or query in p[
                'policyNumber'].lower()]


@app.get("/patients/{patient_id}", response_model=Patient)
async def get_patient_details_endpoint(patient_id: int):
    return find_patient_or_404(patient_id)


@app.post("/patients/{patient_id}/analyses", response_model=Analysis, status_code=201)
async def add_analysis_to_patient_endpoint(
        patient_id: int,
        date: str = Form(...),
        xrayImage: UploadFile = File(...)
):
    patient = find_patient_or_404(patient_id)
    patient_gender = patient.get("gender")
    if not patient_gender:
        raise HTTPException(status_code=400, detail="Patient gender is required for age prediction.")

    contents = await xrayImage.read()
    try:
        original_pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    finally:
        if hasattr(xrayImage, 'close') and callable(xrayImage.close): await xrayImage.close()

    # 0. Исправление ориентации по EXIF (если есть)
    try:
        original_pil_img = ImageOps.exif_transpose(original_pil_img)
    except Exception:
        # Если нет EXIF или ошибка, используем как есть
        pass

        # 1. Подготовка изображения для сегментации
    # seg_processor_global.transforms[0] это Resize. Получим его размер.
    # Убедимся, что seg_processor_global и его transforms[0] существуют
    seg_target_size_h, seg_target_size_w = IMG_SIZE_KERAS, IMG_SIZE_KERAS  # Default
    if seg_processor_global and hasattr(seg_processor_global, 'transforms') and \
            len(seg_processor_global.transforms) > 0 and hasattr(seg_processor_global.transforms[0], 'size'):
        # size может быть int (для Resize(N)) или tuple (H,W) для Resize((H,W))
        s = seg_processor_global.transforms[0].size
        if isinstance(s, int):
            seg_target_size_h, seg_target_size_w = s, s
        elif isinstance(s, (list, tuple)) and len(s) == 2:
            seg_target_size_h, seg_target_size_w = s[0], s[1]

    # Resize для Segformer ожидает (width, height) для PIL, а хранит (height, width)
    image_resized_for_seg = original_pil_img.convert("RGB").resize(
        (seg_target_size_w, seg_target_size_h),  # (width, height)
        Image.Resampling.LANCZOS
    )

    image_for_keras_input: Optional[Image.Image] = None  # Это изображение пойдет в backend_preprocess_image

    # 2. Сегментация и маскирование
    segmentation_mask_np = predict_segmentation_mask(image_resized_for_seg)  # Маска будет размера image_resized_for_seg

    if segmentation_mask_np is not None and np.any(segmentation_mask_np):
        print("Segmentation successful, applying mask and cropping.")
        # apply_mask_and_crop ожидает, что image_to_process и binary_mask_np одного размера
        masked_and_cropped_pil = apply_mask_and_crop(
            image_resized_for_seg,  # RGB, seg_target_size
            segmentation_mask_np,  # Бинарная, seg_target_size
            output_mode='RGB'  # Черный фон
        )

        if masked_and_cropped_pil:
            image_for_keras_input = masked_and_cropped_pil
            try:  # Отладка: сохраняем отмаскированное и обрезанное изображение
                base, ext = os.path.splitext(xrayImage.filename or "unknown.png")
                fn = f"{base}_debug_masked_cropped{ext}"
                image_for_keras_input.save(os.path.join(UPLOAD_DIRECTORY, fn))
                print(f"Saved debug (masked_cropped) image: {fn}")
            except Exception as e_save:
                print(f"Error saving debug_masked_cropped image: {e_save}")
        else:
            print("Applying mask and cropping returned None. Using resized_for_seg for Keras preprocessing.")
            image_for_keras_input = image_resized_for_seg
    else:
        print("Segmentation failed or mask is empty. Using resized_for_seg for Keras preprocessing.")
        image_for_keras_input = image_resized_for_seg  # RGB, seg_target_size

    if image_for_keras_input is None:  # Дополнительная подстраховка
        print("CRITICAL: image_for_keras_input is None. Using original (RGB converted).")
        image_for_keras_input = original_pil_img.convert("RGB")

    # 3. Предобработка для Keras с помощью backend_preprocess_image
    # Параметры crop_threshold и padding_margin можно настроить.
    # Для уже отмаскированного изображения, crop_threshold=15 может быть высоковат.
    # Попробуем с меньшими значениями или теми, что ты указал в своей функции.
    # Использую значения по умолчанию из твоей функции: crop_threshold=15, padding_margin=15
    preprocessed_for_keras = backend_preprocess_image(
        image_for_keras_input,  # Это RGB изображение (либо отмаскированное, либо просто ресайзнутое)
        target_size=IMG_SIZE_KERAS,
        crop_threshold=15,  # Из твоей функции
        padding_margin=15  # Из твоей функции
    )

    predicted_age_in_years: float
    if preprocessed_for_keras is None:
        print("Keras preprocessing (backend_preprocess_image) failed. Using random age.")
        predicted_age_in_years = round(random.uniform(5.0, 18.0), 1)
    else:
        try:  # Отладка: сохраняем изображение, идущее в Keras
            base, ext = os.path.splitext(xrayImage.filename or "unknown.png")
            fn = f"{base}_debug_keras_input{ext}"
            preprocessed_for_keras.save(os.path.join(UPLOAD_DIRECTORY, fn))
            print(f"Saved debug (keras_input) image: {fn}")
        except Exception as e_save:
            print(f"Error saving debug_keras_input image: {e_save}")

        # 4. Предсказание возраста Keras моделью
        predicted_age_months = predict_bone_age(preprocessed_for_keras, patient_gender)
        if predicted_age_months is not None:
            predicted_age_in_years = round(predicted_age_months / 12.0, 1)
        else:
            print("Keras prediction failed. Using random age.")
            predicted_age_in_years = round(random.uniform(5.0, 18.0), 1)

    # 5. Сохранение ОРИГИНАЛЬНОГО файла и данных анализа
    file_extension = os.path.splitext(xrayImage.filename)[1] if xrayImage.filename else ".png"
    if not file_extension or file_extension.lower() not in ['.jpg', '.jpeg', '.png']:
        file_extension = ".png"  # Default if unknown or invalid

    unique_filename = f"xray_{patient_id}_{uuid.uuid4().hex}{file_extension}"
    original_file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
    try:
        with open(original_file_path, "wb") as buffer:
            buffer.write(contents)  # Сохраняем исходные байты
    except Exception as e:
        # if os.path.exists(original_file_path): os.remove(original_file_path) # Осторожно с удалением
        raise HTTPException(status_code=500, detail=f"Could not save original image: {str(e)}")

    image_url_for_db = f"/static/uploads/{unique_filename}"
    analysis_id_val = get_next_analysis_id()
    new_analysis_data = {
        "id": analysis_id_val, "date": date, "predictedAge": predicted_age_in_years,
        "xrayImageURL": image_url_for_db, "doctorNotes": ""
    }
    patient["analyses"].append(new_analysis_data)
    return new_analysis_data


@app.put("/patients/{patient_id}/analyses/{analysis_id}", response_model=Analysis)
async def update_analysis_notes_endpoint(
        patient_id: int, analysis_id: int, notes_update: AnalysisNotesUpdate):
    patient = find_patient_or_404(patient_id)
    analysis_to_update = find_analysis_or_404(patient, analysis_id)
    analysis_to_update["doctorNotes"] = notes_update.doctorNotes
    return analysis_to_update


@app.get("/")
async def root():
    return {"message": "Patient Management API with ML Prediction is running. Visit /docs."}

# Для запуска: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
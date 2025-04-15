from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from process.yeast_classification import classify_yeast_image
from typing import List
import base64
import io
from PIL import Image

yeast_classification_router = APIRouter()

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class BatchImageRequest(BaseModel):
    images: List[str]  # List of base64 encoded images

class ClassificationResponse(BaseModel):
    class_name: str
    confidence: float
    class_id: int

# Hàm decode base64 thành ảnh PIL
def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Lỗi giải mã base64: {e}")

@yeast_classification_router.post("/classify-yeast/single", response_model=ClassificationResponse)
async def classify_single_image(request: ImageRequest):
    """Classify a single yeast cell image"""
    try:
        image = decode_base64_image(request.image)  # Chuyển base64 thành ảnh
        result = classify_yeast_image(image)  # Gửi ảnh vào model
        return ClassificationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@yeast_classification_router.post("/classify-yeast/batch", response_model=List[ClassificationResponse])
async def classify_batch_images(request: BatchImageRequest):
    """Classify multiple yeast cell images"""
    try:
        results = [classify_yeast_image(decode_base64_image(img)) for img in request.images]
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

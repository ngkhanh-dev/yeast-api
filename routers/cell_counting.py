import base64
import numpy as np
from io import BytesIO
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor
from process.Bai_toan_buong_dem import Count_Yeast_in_16_Squares 
from process.prediction import predict_mask_ethanol

yeast_count_router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)






















'''
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import base64
import shutil
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import json
import cv2
sys.path.append(str(Path(__file__).parent.parent))
from process.Bai_toan_buong_dem import Process_with_path, Count_Yeast_in_16_Squares

cell_counting_router = APIRouter()

def save_base64_image(base64_str: str, prefix: str) -> Path:
    """Helper function to save base64 image to temp file"""
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Decode base64 image
    image_data = base64.b64decode(base64_str)
    
    # Save to temp file
    temp_path = temp_dir / f"{prefix}_{os.urandom(8).hex()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_data)
        
    return temp_path

def analyze_squares(image_data: str, image_id: str):
    """Analyze image to detect 16 squares"""
    try:
        # Save base64 image to temp file
        image_path = save_base64_image(image_data, f"squares_{image_id}")
        
        # Process image to get square coordinates
        small_boxes = Process_with_path(str(image_path), show_process=True)
        
        # Convert numpy arrays to lists for JSON
        squares_coords = [box.tolist() for box in small_boxes]
        
        # Prepare response
        response_content = {
            "status": "success",
            "image_id": image_id,
            "squares": [
                {
                    "square_index": i + 1,
                    "coordinates": coords
                } for i, coords in enumerate(squares_coords)
            ]
        }
        
        # Save results
        file_name = os.path.join("saved_json", f"squares_{image_id}.json")
        os.makedirs("saved_json", exist_ok=True)
        with open(file_name, 'w') as json_file:
            json.dump(response_content, json_file)
            
        return response_content
        
    finally:
        # Cleanup
        if 'image_path' in locals():
            try:
                os.remove(image_path)
            except:
                pass

def analyze_yeast(image_data: str, mask_data: str, image_id: str):
    """Analyze images to count yeast in squares"""
    try:
        # Save images to temp files
        image_path = save_base64_image(image_data, f"yeast_{image_id}")
        mask_path = save_base64_image(mask_data, f"mask_{image_id}")
        
        # Process images to count yeast
        processed_image = Count_Yeast_in_16_Squares(str(image_path), str(mask_path))
        
        # Convert processed image back to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response_content = {
            "status": "success",
            "image_id": image_id,
            "processed_image": processed_image_base64
        }
        
        # Save results
        file_name = os.path.join("saved_json", f"yeast_{image_id}.json")
        os.makedirs("saved_json", exist_ok=True)
        with open(file_name, 'w') as json_file:
            json.dump(response_content, json_file)
            
        return response_content
        
    finally:
        # Cleanup
        if 'image_path' in locals():
            try:
                os.remove(image_path)
            except:
                pass
        if 'mask_path' in locals():
            try:
                os.remove(mask_path)
            except:
                pass

executor = ThreadPoolExecutor(max_workers=4)

@cell_counting_router.post("/upload_image/detect-squares")
async def detect_squares(request: Request):
    """Detect 16 squares in an image and return their coordinates"""
    try:
        # Read JSON data from request body
        body = await request.json()
        
        if "base64_image" not in body:
            raise HTTPException(status_code=400, detail="base64_image field is required")
        if "image_id" not in body:
            raise HTTPException(status_code=400, detail="image_id field is required")
            
        base64_image = body["base64_image"]
        image_id = body["image_id"]
        
        future = executor.submit(analyze_squares, base64_image, image_id)
        content = future.result()
        
        return JSONResponse(content=content)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cell_counting_router.post("/upload_image/count-yeast")
async def count_yeast_cells(request: Request):
    """Count yeast cells in each of the 16 squares"""
    try:
        # Read JSON data from request body
        body = await request.json()
        
        if "base64_image" not in body:
            raise HTTPException(status_code=400, detail="base64_image field is required")
        if "base64_mask" not in body:
            raise HTTPException(status_code=400, detail="base64_mask field is required")
        if "image_id" not in body:
            raise HTTPException(status_code=400, detail="image_id field is required")
            
        base64_image = body["base64_image"]
        base64_mask = body["base64_mask"]
        image_id = body["image_id"]
        
        future = executor.submit(analyze_yeast, base64_image, base64_mask, image_id)
        content = future.result()
        
        return JSONResponse(content=content)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
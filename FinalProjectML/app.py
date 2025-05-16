from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ai.inference import predict_from_features
from pathlib import Path
import uuid
from pydantic import BaseModel
import subprocess
import os
import time
import numpy as np

# Cấu hình thư mục gốc và static
WORK_DIR = Path("M:\ML\FinalProjectML")
STATIC_DIR = WORK_DIR / "static"
TEMP_DIR = STATIC_DIR / "temp"

app = FastAPI(docs_url=None, redoc_url=None)

# Mount static chuẩn để truy cập file trong /static/temp
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=WORK_DIR / "templates")

# Cho phép gọi API từ mọi nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dữ liệu user mẫu
fake_users = {
    "admin": {"password": "1", "position": "admin"},
    "guest": {"password": "1", "position": "lecture"},
}

# Trang chủ
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Trang deepfake
@app.get("/doctorcuong", response_class=HTMLResponse)
async def doctorcuong(request: Request):
    return templates.TemplateResponse("doctorcuong.html", {"request": request})

# Đăng nhập
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username in fake_users and fake_users[username]["password"] == password:
        return {
            "success": True,
            "position": fake_users[username]["position"]
        }
    return {
        "success": False,
        "message": "Sai tên đăng nhập hoặc mật khẩu"
    }

# Schema input 13 đặc trưng
class HeartInput(BaseModel):
    features: list[float]
    
@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    f1: str = Form("0"), f2: str = Form("0"), f3: str = Form("0"),
    f4: str = Form("0"), f5: str = Form("0"), f6: str = Form("0"),
    f7: str = Form("0"), f8: str = Form("0"), f9: str = Form("0"),
    f10: str = Form("0"), f11: str = Form("0"), f12: str = Form("0"),
    f13: str = Form("0"),
):
    # Chuyển các giá trị từ str sang float, nếu trống mặc định là "0"
    features = []
    for val in [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13]:
        try:
            features.append(float(val))
        except:
            features.append(0.0)
    
    features_array = np.array(features).reshape(1, -1)
    
    result = predict_from_features(features_array)
    
    if result == False:
        return templates.TemplateResponse("predict.html", {"request": request, "result": "Result: NOT at risk of heart disease"})
    else:
        return templates.TemplateResponse("predict.html", {"request": request, "result": "Result: At risk of heart disease"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

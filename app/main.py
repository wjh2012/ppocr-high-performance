import cv2
import numpy as np
from fastapi import  FastAPI
from starlette.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from fastapi import UploadFile, File

from app.ocr_response import OcrResponse, convert_ocr_result_to_response_fields

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_engine = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_detection_model_dir="models/det/PP-OCRv5_mobile_det_infer",
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    text_recognition_model_dir="models/rec/korean_PP-OCRv5_mobile_rec_infer",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_hpi=True,
)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "이미지 디코딩 실패"}

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = ocr_engine.predict(input=image)[0]
    fields = convert_ocr_result_to_response_fields(result)

    return OcrResponse(inferResult="", message="", fields={"0": fields})
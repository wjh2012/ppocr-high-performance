FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN paddleocr install_hpi_deps gpu

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
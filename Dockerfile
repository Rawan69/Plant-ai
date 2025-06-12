# نبدأ من نسخة Python فيها distutils شغالة
FROM python:3.10-slim

# نحدد مجلد العمل داخل الحاوية
WORKDIR /app

# ننسخ الملفات من جهازك إلى الحاوية
COPY requirements.txt .
COPY final2.py .

# نثبّت المكتبات المطلوبة
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# نفتح البورت اللي Flask بيشتغل عليه (عادة 5000)
EXPOSE 5000

# نحدد الأمر اللي يشغّل السيرفر
CMD ["python", "final2.py"]

RUN apt-get update && apt-get install -y python3-distutils

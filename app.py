import os
import random
import time
import csv

from flask import Flask, render_template, request, redirect, session, url_for
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from openai import OpenAI

# Load OpenAI client with secret key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Flask setup
app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Configs
PHISH_FOLDER = 'static/images/PHISH'
LEGIT_FOLDER = 'static/images/LEGIT'
UPLOAD_FOLDER = 'static/uploads'
TOTAL_QUESTIONS = 10

# Tesseract config
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# -------- OCR Processing --------
def preprocess_image_for_ocr(image):
    image = image.convert('L')  # grayscale
    image = image.filter(ImageFilter.MedianFilter())
    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    return image

# -------- GPT Classifier --------
def classify_with_gpt(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert. Based on the email content, decide if it is PHISH or LEGIT. Only respond with 'PHISH' or 'LEGIT'."
                },
                {
                    "role": "user",
                    "content": f"Email content:\n{text.strip()}\n\nIs this PHISH or LEGIT?"
                }
            ]
        )
        message = response.choices[0].message.content.strip().upper()
        print("✅ GPT Response:", message)
        if message not in ["PHISH", "LEGIT"]:
            return "ERROR"
        return message
    except Exception as e:
        print("❌ GPT Error:", e)
        return "ERROR"

# -------- Routes --------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/quiz-start')
def index():
    session['score'] = 0
    session['ai_score'] = 0
    session['round'] = 0
    session['questions'] = []
    session['start_time'] = time.time()
    session['ai_time'] = 0
    session['history'] = []

    phish_images = os.listdir(PHISH_FOLDER)
    legit_images = os.listdir(LEGIT_FOLDER)
    combined = [('PHISH', img) for img in phish_images] + [('LEGIT', img) for img in legit_images]
    random.shuffle(combined)
    session['questions'] = combined[:min(TOTAL_QUESTIONS, len(combined))]

    return redirect(url_for('quiz'))

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if request.method == 'POST':
        if session['round'] == 0 or session['round'] > len(session['questions']):
            return redirect(url_for('result'))

        user_answer = request.form.get('answer')
        real_label, filename = session['questions'][session['round'] - 1]
        start_ai = time.time()

        if user_answer == real_label:
            session['score'] += 1

        image_path = os.path.join('static/images', real_label, filename)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"❌ Error loading image: {image_path} -> {e}")
            return redirect(url_for('result'))

        image = preprocess_image_for_ocr(image)
        text = pytesseract.image_to_string(image, config='--psm 6')

        gpt_label = classify_with_gpt(text)

        end_ai = time.time()
        session['ai_time'] += end_ai - start_ai

        if gpt_label == "ERROR":
            session['history'].append({
                "image": filename,
                "real": real_label,
                "human": user_answer,
                "ai": "ERROR",
                "qr_urls": []
            })
            return redirect(url_for('result'))

        if gpt_label == real_label:
            session['ai_score'] += 1

        session['history'].append({
            "image": filename,
            "real": real_label,
            "human": user_answer,
            "ai": gpt_label,
            "qr_urls": []
        })

        with open('results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, real_label, user_answer, gpt_label,
                             user_answer == real_label, gpt_label == real_label])

    if session['round'] >= len(session['questions']):
        return redirect(url_for('result'))

    label, filename = session['questions'][session['round']]
    session['round'] += 1
    image_path = f'images/{label}/{filename}'

    return render_template('quiz.html', image=image_path, round=session['round'], total=len(session['questions']))

@app.route('/result')
def result():
    human_time = time.time() - session['start_time']
    return render_template('result.html',
                           score=session['score'],
                           ai_score=session['ai_score'],
                           total=len(session['questions']),
                           human_time=round(human_time, 2),
                           ai_time=round(session['ai_time'], 2),
                           history=session['history'])

@app.route('/ocr-test', methods=['GET', 'POST'])
def ocr_test():
    extracted_text = ""
    if request.method == 'POST':
        file = request.files['image']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            image = Image.open(path)
            image = preprocess_image_for_ocr(image)
            extracted_text = pytesseract.image_to_string(image, config='--psm 6')
    return render_template('ocr_test.html', extracted_text=extracted_text)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import re
import nltk
from enchant.checker import SpellChecker
import torch
from transformers import BertTokenizer, BertForMaskedLM
from difflib import SequenceMatcher
import tempfile

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function definitions remain unchanged
def assess_noise_type(image):
    std_dev = np.std(image)
    if std_dev < 50:
        return 'Salt and Pepper'
    elif std_dev > 50 and std_dev < 100:
        return 'Gaussian'
    elif std_dev > 100:
        return 'Speckle'
    else:
        return 'Unknown'

def auto_denoise(image):
    noise_type = assess_noise_type(image)
    if noise_type == 'Gaussian':
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    elif noise_type == 'Salt and Pepper':
        denoised_image = cv2.medianBlur(image, 5)
    elif noise_type == 'Speckle':
        denoised_image = cv2.medianBlur(image, 3)
    else:
        denoised_image = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised_image

def dynamic_sharpen(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient /= gradient.max()
    gradient_resized = cv2.resize(gradient, (image.shape[1], image.shape[0]))
    min_strength = 0.5
    max_strength = 2.0
    sharpen_strength = min_strength + (max_strength - min_strength) * gradient_resized
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = np.zeros_like(image, dtype=np.float64)
    for i in range(3):
        sharpened_image = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened_image, 0, 255).astype(np.uint8)

def equalize_histogram_color(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    equalized_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return equalized_image

def adjust_brightness_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    adjusted_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return adjusted_image

def name_persons_list(texto) -> list:
    nameslist = []
    for sentence in nltk.sent_tokenize(texto):
        for tags in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence))):
            if isinstance(tags, nltk.tree.Tree) and tags.label() == 'PERSON':
                nameslist.insert(0, (tags.leaves()[0][0]))
    return list(set(nameslist))

def pred_correct_words(original_text, output_pred, IDS_FOW, recommended_words) -> str:
    tk = BertTokenizer.from_pretrained('bert-large-uncased')
    for index in range(len(IDS_FOW)):
        if index >= len(recommended_words):
            break
        predictions = torch.topk(output_pred[0, IDS_FOW[index]], k=50)
        inds = predictions.indices.tolist()
        ltk1 = tk.convert_ids_to_tokens(inds)
        ltk2 = recommended_words[index]
        max_sims = 0
        tokens_pred = ''
        for ws1 in ltk1:
            for ws2 in ltk2:
                sequencem = SequenceMatcher(None, ws1, ws2).ratio()
                if sequencem is not None and sequencem > max_sims:
                    max_sims = sequencem
                    tokens_pred = ws1
        original_text = original_text.replace('[MASK]', tokens_pred, 1)
    return original_text

st.title('Image OCR and Text Correction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")

    denoised_image = auto_denoise(image_np)
    sharpened_image = dynamic_sharpen(denoised_image)
    equalized_image = equalize_histogram_color(sharpened_image)
    adjusted_image = adjust_brightness_contrast(equalized_image)
    gray_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Save the processed image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        cv2.imwrite(temp_file.name, equalized_image)
        image_path = temp_file.name

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image_path, cls=True)

    boxes = [line[0] for line in result[0]]
    texts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    # Text preprocessing
    regexp = {
        '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ', '"': ' " ', '"': ' " ', ',': ' , ', '.': ' . ', '!': ' ! ',
        '?': ' ? ', "n't": " not", "'ll": " will", '*': ' * ', '(': ' ( ', ')': ' ) ', "s'": "s '"
    }
    regexp = dict((re.escape(k), v) for k, v in regexp.items())
    pat_recog_tok = re.compile("|".join(regexp.keys()))

    aux_text = [pat_recog_tok.sub(lambda m: regexp[re.escape(m.group(0))], text) for text in texts]
    phras = nltk.sent_tokenize(' '.join(aux_text))

    nameslist = name_persons_list(' '.join(aux_text))
    words_to_ignore = nameslist + ["!", ",", ".", "\"", "?", '(', ')', '*', "'"]

    spellcheck = SpellChecker("en_US")
    sep_w = ' '.join(aux_text).split()

    anomaly_words = [word for word in sep_w if not spellcheck.check(word) and word not in words_to_ignore]
    recommended_words = [spellcheck.suggest(word) for word in anomaly_words]

    aux_text_str = ' '.join(aux_text)
    original_text = aux_text_str
    for word in anomaly_words:
        aux_text_str = aux_text_str.replace(word, '[MASK]')
        original_text = original_text.replace(word, '[MASK]')

    tk = BertTokenizer.from_pretrained('bert-large-uncased')
    text_toknd = tk.tokenize(aux_text_str)
    idx_toknd = tk.convert_tokens_to_ids(text_toknd)
    IDS_FOW = [idx for idx, wd in enumerate(text_toknd) if wd == '[MASK]']

    segments = [idx for idx, wd in enumerate(text_toknd) if wd == "."]
    ids_seg = []
    w_prior = -1

    for key, sentnc in enumerate(segments):
        ids_seg += [key] * (sentnc - w_prior)
        w_prior = sentnc

    ids_seg += [len(segments)] * (len(text_toknd) - len(ids_seg))
    sgms_tensor = torch.tensor([ids_seg])
    tkns_tensor = torch.tensor([idx_toknd])

    with torch.no_grad():
        model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        output_pred = model(tkns_tensor, sgms_tensor).logits

    original_text = pred_correct_words(original_text, output_pred, IDS_FOW, recommended_words)

    # Display results
    st.write("\n=========== OCR RESULTS ================")
    for text, score in zip(texts, scores):
        st.write(f'{text} - {score:.2f}')

    st.write("\n=========== ANOMALY WORDS ================")
    st.write(anomaly_words)

    st.write("\n=========== RECOMMENDED WORDS ================")
    for word_list in recommended_words:
        st.write(word_list)

    st.write("\n=========== MASKED TEXT ================")
    st.write(aux_text_str)

    st.write("\n=========== ORIGINAL TEXT WITH PREDICTED WORDS ================")
    st.write(original_text)

import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

def correct_image_orientation(pil_img):
    cv_img = np.array(pil_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    rotated_pil = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    return rotated_pil

st.title("📷 TextifyImages")

uploaded_files = st.file_uploader("画像ファイルを選択（複数可）", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        corrected_img = correct_image_orientation(image)
        st.image(corrected_img, caption=f"補正済み画像 {i+1}", use_column_width=True)
        text = pytesseract.image_to_string(corrected_img, lang='jpn+eng')
        all_text += f"\n--- 画像{i+1} ---\n{text}\n"

    st.subheader("📝 抽出されたテキスト")
    st.text_area("結果", all_text, height=400)
    st.download_button("📄 テキストをダウンロード", data=all_text, file_name="ocr_result.txt", mime="text/plain")

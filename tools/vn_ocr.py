import os
import cv2
import easyocr
import numpy as np
import re

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        print("⏳ Đang khởi tạo EasyOCR (Final Polish v2)...")
        _reader = easyocr.Reader(['vi', 'en'], gpu=True)
        print("✅ EasyOCR sẵn sàng!")
    return _reader

def post_process_vietnamese(text):
    # HẬU XỬ LÝ 1: OCR hay nhầm chữ 'I' in hoa thành 'l' in thường (vd: TỘl -> TỘI). Dùng Regex để ép chữ 'l' đứng sau ký tự in hoa thành chữ 'I'.
    text = re.sub(r'([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ])l\b', r'\1I', text)
    
    # HẬU XỬ LÝ 2: Sử dụng bộ Từ điển Hard-code để sửa các lỗi nhận diện "ảo giác" của thư viện đối với tiếng Việt có dấu.
    text = text.replace("@UẨN", "QUÂN").replace("sẢM", "SẢN").replace("TAI", "TÀI")
    text = text.replace("CỐÝ", "CỐ Ý").replace("TUHINH", "TỬ HÌNH")
    
    replacements = {
        "GIET NGUOI": "GIẾT NGƯỜI",
        "SỬ DUNG": "SỬ DỤNG",
        "TRẤI PHÉP": "TRÁI PHÉP",
        "TỘL": "TỘI",
        "ĐỄ": "ĐỀ",
        "VÊ": "VỀ",
        "VỄ": "VỀ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'(\d)(\d)\s+NĂM', r'\1-\2 NĂM', text)
    return text

def extract_text_from_image(image_source):
    """
    image_source: Có thể là đường dẫn (string) hoặc mảng numpy (cv2 image)
    """
    if isinstance(image_source, str):
        if not os.path.exists(image_source): return ""
        img = cv2.imread(image_source)
    else:
        img = image_source # Giả định là numpy array từ clipboard
        
    if img is None: return ""

    # TIỀN XỬ LÝ ẢNH BẰNG OPENCV: Làm nét và chuẩn hóa ảnh mờ/nén từ Facebook/Zalo
    # 1. Resize: Phóng to ảnh lên 2.5 lần để chữ sắc nét hơn (dùng nội suy INTER_CUBIC)
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    # 2. Grayscale: Chuyển ảnh màu sang xám để giảm tải tính toán
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. CLAHE: Cân bằng sáng tối cục bộ (cứu những ảnh bị chói sáng một góc hoặc tối mù)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    # 4. Invert: Đảo ngược màu (đen thành trắng, trắng thành đen)
    inverted = cv2.bitwise_not(contrast)
    # 5. Otsu Threshold: Nhị phân hóa tuyệt đối. Tẩy trắng nền, làm nét chữ đen mượt mà, khử sạch nhiễu hạt muối tiêu.
    _, thresh = cv2.threshold(inverted, 160, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    os.makedirs("output", exist_ok=True)
    temp_path = "output/ocr_current.jpg"
    cv2.imwrite(temp_path, thresh)

    reader = get_reader()
    # Gọi EasyOCR đọc chữ. Tham số paragraph=True tự động gom cụm (cluster) các dòng chữ có khoảng cách gần và thẳng lề thành một Đoạn Văn duy nhất.
    results = reader.readtext(temp_path, detail=1, paragraph=True, contrast_ths=0.05)

    content_lines = []
    for (bbox, text) in results:
        text = post_process_vietnamese(text.strip())
        if len(text) > 4: content_lines.append(text)

    return " ".join(content_lines)

if __name__ == "__main__":
    p = input("Path: ").strip('"')
    print("\nResult:", extract_text_from_image(p))
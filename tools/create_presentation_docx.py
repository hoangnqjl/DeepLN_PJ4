import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_element(name):
    return OxmlElement(name)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def build_docx():
    doc = Document()
    
    # Page setup - Margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles Setup
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Arial'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x33, 0x33, 0x33) # Charcoal
    
    # Colors
    color_primary = RGBColor(0x1B, 0x36, 0x5D)    # Deep Navy
    color_secondary = RGBColor(0x00, 0x80, 0x80)  # Teal
    color_dark = RGBColor(0x22, 0x22, 0x22)
    
    # 1. Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("BÁO CÁO NGẮN GỌN VỀ MÔ HÌNH PHÂN LOẠI TIN GIẢ")
    title_run.font.name = 'Arial'
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.color.rgb = color_primary
    title.paragraph_format.space_after = Pt(24)
    
    # Subtitle
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle.add_run("So sánh hai kiến trúc tiêu biểu: PhoBERT và LSTM")
    sub_run.font.name = 'Arial'
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    subtitle.paragraph_format.space_after = Pt(36)
    
    # Divider
    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_div = p_div.add_run("_______________________________________________________________")
    run_div.font.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    p_div.paragraph_format.space_after = Pt(24)
    
    # 2. Part 1: PhoBERT
    h1 = doc.add_paragraph()
    h1_run = h1.add_run("1. Mô hình PhoBERT (Vietnamese RoBERTa)")
    h1_run.font.name = 'Arial'
    h1_run.font.size = Pt(14)
    h1_run.font.bold = True
    h1_run.font.color.rgb = color_primary
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    
    p1 = doc.add_paragraph()
    p1.add_run(
        "PhoBERT là mô hình ngôn ngữ lớn (Pre-trained Language Model) được tối ưu hóa riêng cho Tiếng Việt bởi VinAI Research dựa trên kiến trúc RoBERTa (một nhánh cải tiến mạnh mẽ của BERT từ Meta). Mô hình nổi bật nhờ khả năng hiểu sâu sắc cấu trúc ngữ nghĩa tiếng Việt bản địa."
    )
    p1.paragraph_format.space_after = Pt(12)
    
    # Bullet points for PhoBERT
    bullet1_1 = doc.add_paragraph(style='List Bullet')
    r1 = bullet1_1.add_run("Kiến trúc cốt lõi: ")
    r1.bold = True
    bullet1_1.add_run("Sử dụng cơ chế Self-Attention của Transformer, xử lý toàn bộ các từ trong câu cùng lúc để nắm bắt ngữ cảnh hai chiều một cách toàn diện.")
    
    bullet1_2 = doc.add_paragraph(style='List Bullet')
    r2 = bullet1_2.add_run("Tách từ Tiếng Việt: ")
    r2.bold = True
    bullet1_2.add_run("Kết hợp cùng công cụ tách từ chuyên sâu (PyVi) giúp phân tách chính xác các từ ghép đơn lập đặc thù trong tiếng Việt trước khi Tokenize BPE.")
    
    bullet1_3 = doc.add_paragraph(style='List Bullet')
    r3 = bullet1_3.add_run("Phương pháp huấn luyện: ")
    r3.bold = True
    bullet1_3.add_run("Sử dụng cơ chế Đóng băng trọng số (Feature Extraction) cho bộ dữ liệu data_01 và Tinh chỉnh hiệu quả tham số (LoRA - Low-Rank Adaptation) tích hợp Mixed Precision FP16 cho bộ dữ liệu data_03.")
    
    # 3. Part 2: LSTM
    h2 = doc.add_paragraph()
    h2_run = h2.add_run("2. Mô hình LSTM (Long Short-Term Memory)")
    h2_run.font.name = 'Arial'
    h2_run.font.size = Pt(14)
    h2_run.font.bold = True
    h2_run.font.color.rgb = color_primary
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)
    
    p2 = doc.add_paragraph()
    p2.add_run(
        "LSTM là mạng hồi quy (RNN - Recurrent Neural Network) cải tiến, được thiết kế chuyên biệt để xử lý dữ liệu dạng chuỗi tuần tự như ngôn ngữ tự nhiên. LSTM giải quyết triệt để vấn đề triệt tiêu đạo hàm bằng cách tích hợp tế bào bộ nhớ và cơ chế cổng."
    )
    p2.paragraph_format.space_after = Pt(12)
    
    # Bullet points for LSTM
    bullet2_1 = doc.add_paragraph(style='List Bullet')
    rb1 = bullet2_1.add_run("Xử lý chuỗi tuần tự: ")
    rb1.bold = True
    bullet2_1.add_run("Đọc văn bản tuần tự từng từ một để mô hình hóa mối quan hệ thời gian và thứ tự giữa các từ trong câu.")
    
    bullet2_2 = doc.add_paragraph(style='List Bullet')
    rb2 = bullet2_2.add_run("Cơ chế ba loại cổng: ")
    rb2.bold = True
    bullet2_2.add_run("Sử dụng Cổng Quên (Forget Gate), Cổng Nhớ (Input Gate) và Cổng Xuất (Output Gate) để chủ động giữ lại các thông tin quan trọng lâu dài và loại bỏ thông tin dư thừa.")
    
    bullet2_3 = doc.add_paragraph(style='List Bullet')
    rb3 = bullet2_3.add_run("Hiệu năng tối ưu: ")
    rb3.bold = True
    bullet2_3.add_run("Kích thước mô hình gọn nhẹ, số lượng tham số nhỏ hơn hàng trăm lần so với PhoBERT, giúp huấn luyện cực kỳ nhanh chóng trên cả CPU.")
    
    # 4. Part 3: Comparison Table
    h3 = doc.add_paragraph()
    h3_run = h3.add_run("3. Bảng so sánh tổng quan giữa hai mô hình")
    h3_run.font.name = 'Arial'
    h3_run.font.size = Pt(14)
    h3_run.font.bold = True
    h3_run.font.color.rgb = color_primary
    h3.paragraph_format.space_before = Pt(18)
    h3.paragraph_format.space_after = Pt(12)
    
    # Add Table
    table = doc.add_table(rows=6, cols=3)
    table.style = 'Light Shading Accent 1'
    
    # Table headers
    headers = ["Tiêu chí so sánh", "PhoBERT (Transformer)", "LSTM (Mạng hồi quy RNN)"]
    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = h_text
        set_cell_background(hdr_cells[i], "1B365D") # Navy Background
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=180, right=180)
        # Style header text to white bold
        for p in hdr_cells[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(10.5)
                
    # Data rows
    data = [
        ["Cơ chế cốt lõi", "Self-Attention (Song song hóa toàn bộ)", "Cổng nhớ & quên (Xử lý tuần tự từng từ)"],
        ["Huấn luyện sẵn", "Đã pre-trained trên 20GB văn bản Việt", "Tự học hoàn toàn từ đầu trên tập train"],
        ["Độ hiểu ngữ nghĩa", "Cực kỳ sâu sắc, hiểu rõ từ lóng và ẩn ý", "Cơ bản - Khá, nắm bắt tốt cấu trúc tuần tự"],
        ["Tài nguyên phần cứng", "Đòi hỏi GPU mạnh (LoRA giúp giảm nhẹ)", "Cực nhẹ, chạy tốt ngay cả trên CPU"],
        ["Tốc độ huấn luyện", "Chậm (Yêu cầu tài nguyên tính toán lớn)", "Nhanh vượt trội (Hoàn thành trong vài phút)"]
    ]
    
    for row_idx, row_data in enumerate(data):
        row_cells = table.rows[row_idx + 1].cells
        for col_idx, text in enumerate(row_data):
            row_cells[col_idx].text = text
            set_cell_margins(row_cells[col_idx], top=100, bottom=100, left=150, right=150)
            if row_idx % 2 == 1:
                set_cell_background(row_cells[col_idx], "F4F6F9") # Zebra light gray
            # Format text
            for p in row_cells[col_idx].paragraphs:
                if col_idx == 0:
                    for run in p.runs:
                        run.font.bold = True
                        run.font.size = Pt(10)
                else:
                    for run in p.runs:
                        run.font.size = Pt(10)
                        
    # Footer info
    p_foot = doc.add_paragraph()
    p_foot.paragraph_format.space_before = Pt(36)
    p_foot.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    f_run = p_foot.add_run("Báo cáo được khởi tạo tự động phục vụ Nghiên cứu dự án DeepLN.")
    f_run.font.size = Pt(9)
    f_run.font.italic = True
    f_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    
    # Ensure docs directory exists
    os.makedirs(os.path.join(".", "docs"), exist_ok=True)
    out_path = os.path.join(".", "docs", "gioi_thieu_mo_hinh.docx")
    doc.save(out_path)
    print(f"Document successfully created and saved to: {out_path}")

if __name__ == "__main__":
    build_docx()

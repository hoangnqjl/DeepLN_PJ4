# Khung Lý Thuyết Dự Án Nhận Diện Tin Giả (Fake News Detection)

Tài liệu này tổng hợp các kiến thức cốt lõi về các tham số, kiến trúc mô hình và quy trình hệ thống được sử dụng trong dự án, kết hợp giữa lý thuyết hàn lâm và kinh nghiệm thực tiễn.

---

## 1. Các Tham Số Quan Trọng (Hyperparameters)

### 1.1. Batch Size (Kích thước lô)
*   **Định nghĩa:** Số lượng mẫu dữ liệu được đưa vào mô hình trong một lần tính toán.
*   **Ý nghĩa:** 
    *   **Nhỏ (8, 16):** Giúp mô hình thoát khỏi các điểm tối ưu cục bộ, nhưng huấn luyện lâu hơn.
    *   **Lớn (32):** Tính toán ổn định, tận dụng sức mạnh GPU nhưng tốn VRAM.

### 1.2. Dropout (Tỷ lệ bỏ rơi)
*   **Cơ chế:** Ngẫu nhiên loại bỏ một tỷ lệ neuron trong quá trình train để chống **Overfitting**.
*   **Ý nghĩa:** Buộc các neuron phải học độc lập, không dựa dẫm vào nhau (giống như việc xoay vòng cầu thủ trong đội bóng để ai cũng giỏi).
*   **Lưu ý:** Nếu `Dropout` quá cao (ví dụ 0.9), mô hình sẽ rơi vào trạng thái **Underfitting** (không học được gì) do lượng thông tin bị mất mát quá lớn.

### 1.3. Learning Rate (Tốc độ học)
*   **Ý nghĩa:** Độ dài của mỗi "bước đi" khi tìm điểm tối ưu. 
*   **Chiến lược:** 
    *   **LSTM (1e-3):** Bước đi lớn hơn vì học từ đầu.
    *   **PhoBERT (2e-5):** Bước đi rất nhỏ để thực hiện **Fine-tuning**, tránh phá hỏng kiến thức ngôn ngữ khổng lồ đã học sẵn.

---

## 2. Cấu Trúc Chi Tiết Mô Hình LSTM (Long Short-Term Memory)

Mô hình học theo trình tự (Sequential), phù hợp xử lý dữ liệu chuỗi thời gian.

1.  **Tầng Embedding:** Chuyển ID từ vựng thành vector 100 chiều. Đây là dạng **Static Embedding** (từ điển tĩnh).
2.  **Hai Trạng Thái Cốt Lõi:**
    *   **Cell State ($c$) - Trí nhớ dài hạn:** Giống như một **"Cuốn sổ tay ghi chép"** xuyên suốt câu văn, lưu giữ các manh mối quan trọng nhất và loại bỏ rác thông qua Forget/Input gates.
    *   **Hidden State ($h$) - Trí nhớ ngắn hạn:** Giống như **"Ý nghĩ tức thời"**, thay đổi sau mỗi từ để đưa ra đầu ra và dẫn dắt từ tiếp theo.
3.  **Số lớp (n_layers=2):** Việc xếp chồng 2 lớp giúp mô hình học được các đặc trưng phức tạp hơn (lớp 1 học từ loại, lớp 2 học ngữ nghĩa).

---

## 3. Cấu Trúc Chi Tiết Mô Hình PhoBERT (Transformer)

Mô hình hiện đại sử dụng cơ chế Attention để nhìn toàn bộ văn bản cùng lúc.

1.  **Self-Attention:** Giúp mỗi từ "chú ý" đến tất cả các từ khác để hiểu ngữ cảnh.
2.  **Positional Encoding (Mã hóa vị trí):** Vì nhìn cả câu cùng lúc nên PhoBERT cần cộng thêm vector vị trí để biết thứ tự trước sau của các từ.
3.  **Token đặc biệt `<s>`:** Đóng vai trò là **"Người đại diện lớp"**. Nó tổng hợp thông tin từ toàn bộ các từ trong câu để đưa vào lớp phân loại cuối cùng.
4.  **Dynamic Embedding:** Giúp PhoBERT hiểu từ theo hoàn cảnh (ví dụ: phân biệt được từ "Hay" là khen thật hay là mỉa mai dựa vào các từ xung quanh).

---

## 4. Quy Trình Hệ Thống (Pipeline)

### 4.1. Tiền xử lý (Structuring Data)
1.  **Làm sạch:** Xóa URL, ký tự đặc biệt, đưa về chữ thường.
2.  **Tách từ (ViTokenizer):** Nối từ ghép tiếng Việt (ví dụ: `thanh_tra`).
3.  **Padding (<PAD>):** Lấp đầy các câu ngắn cho bằng `max_len`. 
    *   *Lưu ý:* Dù `<PAD>` vô nghĩa nhưng máy tính vẫn phải thực hiện phép nhân ma trận trên nó, nên `max_len` càng lớn thì máy chạy càng chậm.

### 4.2. Huấn luyện và Tối ưu (Training & Optimization)
*   **Fine-tuning:** Tinh chỉnh mô hình PhoBERT trên tập tin giả.
*   **Freezing:** Kỹ thuật đóng băng các lớp cũ (`requires_grad = False`) để chỉ dạy kiến thức mới cho lớp phân loại cuối cùng.
*   **Class Weights:** "Phạt nặng" mô hình khi đoán sai tin giả (lớp thiểu số) để cân bằng lại dữ liệu bị lệch.
*   **Optimization:** Sử dụng **Adam/AdamW** để tự động điều chỉnh bước đi của quá trình học.

---

## 5. Đánh Giá Và Chỉ Số Chất Lượng

Khi dữ liệu bị mất cân bằng (tin thật nhiều hơn tin giả), ta không chỉ nhìn vào **Accuracy**:
*   **Precision (Độ chính xác):** Ưu tiên khi muốn tránh việc "báo giả sai" (Tránh gây kiện tụng).
*   **Recall (Độ triệu hồi):** Ưu tiên khi muốn "không bỏ sót tin giả" (Tránh loạn xã hội).
*   **F1-Score:** Sự cân bằng hoàn hảo nhất. Nếu một trong hai chỉ số trên thấp, F1-Score sẽ bị kéo xuống rất mạnh.

---
*Tài liệu tổng hợp phục vụ thi vấn đáp và bảo vệ đồ án - Cập nhật 06/05/2026.*
) | Toàn cục (Global - nhìn thấy mọi từ cùng lúc) |
---

## 6. Quy Trình Xử Lý (Pipeline) Chi Tiết

Dưới đây là luồng hoạt động từ dữ liệu thô cho đến khi đưa ra kết quả cuối cùng:

### 5.1. Text (Dữ liệu văn bản thô)
*   Đầu vào là các bài báo hoặc đoạn tin nhắn tiếng Việt.
*   Dữ liệu được làm sạch cơ bản (loại bỏ ký tự đặc biệt, đưa về chữ thường).

### 5.2. Tokenization (Tách từ & Mã hóa)
Đây là bước chuyển đổi ngôn ngữ con người sang con số mà máy tính hiểu được:
*   **Với LSTM:** Sử dụng bộ từ điển tự xây dựng. Mỗi từ được gán một số nguyên (ID). Nếu câu ngắn hơn 100 từ sẽ được thêm các token `<PAD>` (bù trừ), nếu dài hơn sẽ bị cắt bớt.
*   **Với PhoBERT:** Sử dụng bộ tách từ BPE (Byte Pair Encoding) chuyên cho tiếng Việt. Thêm các token đặc biệt như `<s>` (bắt đầu câu) và `</s>` (kết thúc câu).

### 5.3. Model (Kiến trúc mô hình)
*   Khởi tạo cấu trúc mạng (LSTM) hoặc tải trọng số đã học sẵn (PhoBERT) vào bộ nhớ (RAM/GPU).
*   Thiết lập các tham số đầu ra phù hợp với bài toán 2 nhãn (Binary Classification).

### 5.4. Fine-tuning (Huấn luyện / Tinh chỉnh)
*   Đưa dữ liệu đã mã hóa vào mô hình theo từng đợt (Batch).
*   Sử dụng hàm mất mát (**Cross Entropy Loss**) để đo lường sai số.
*   Sử dụng bộ tối ưu (**Adam**) để cập nhật trọng số dựa trên sai số đó.
*   Quá trình này lặp lại qua nhiều vòng (**Epochs**) cho đến khi mô hình đạt độ chính xác mong muốn.

### 5.5. Prediction (Dự đoán)
*   Khi có một đoạn tin văn bản mới, nó đi qua bước 2 (Tokenization) rồi nạp vào mô hình đã huấn luyện xong.
*   Mô hình trả về các con số (Logits). Ta dùng hàm **Argmax** để chọn ra nhãn có xác suất cao nhất (0 hoặc 1).

### 5.6. Evaluation (Đánh giá)
Sau khi dự đoán, ta cần các thước đo để biết mô hình tốt đến đâu:
*   **Accuracy (Độ chính xác):** Tỷ lệ đoán đúng trên tổng số mẫu.
*   **Precision (Độ chính xác trên lớp dự đoán):** Trong những tin mô hình bảo là "Giả", có bao nhiêu tin thực sự là "Giả".
*   **Recall (Độ triệu hồi):** Trong tất cả các tin "Giả" thực tế, mô hình tìm ra được bao nhiêu tin.
*   **F1-Score:** Giá trị trung bình điều hòa giữa Precision và Recall. Đây là thước đo quan trọng nhất khi dữ liệu bị lệch (số tin thật nhiều hơn tin giả hoặc ngược lại).

---

## 7. Kết Luận

Dự án đã triển khai thành công hệ thống nhận diện tin giả (Fake News Detection) cho văn bản tiếng Việt, áp dụng cả hai kiến trúc học sâu phổ biến là LSTM và PhoBERT. Qua quá trình thực nghiệm, có thể rút ra một số kết luận:
*   **Hiệu năng mô hình:** PhoBERT, nhờ cơ chế Self-Attention và kiến thức ngôn ngữ học sẵn (pre-trained), thể hiện sự vượt trội trong việc nắm bắt ngữ cảnh phức tạp của tiếng Việt. LSTM tuy có lợi thế về tốc độ huấn luyện và triển khai nhẹ nhàng nhưng bị hạn chế khi xử lý các câu văn quá dài.
*   **Quy trình chuẩn hóa:** Việc xây dựng một pipeline chuẩn từ bước làm sạch dữ liệu, mã hóa (Tokenization) đến tinh chỉnh (Fine-tuning) giúp hệ thống hoạt động ổn định và có thể tái sử dụng dễ dàng.
*   **Xử lý dữ liệu:** Các kỹ thuật xử lý dữ liệu mất cân bằng (như điều chỉnh Class Weights) đóng vai trò quyết định giúp mô hình không bị thiên lệch, đảm bảo độ tin cậy của chỉ số F1-Score.

---

## 8. Hướng Phát Triển (Future Work)

Để nâng cao độ chính xác và tính ứng dụng thực tiễn, hệ thống có thể được cải tiến theo các hướng sau:
1.  **Mở rộng tập dữ liệu:** Thu thập thêm dữ liệu từ nhiều lĩnh vực khác nhau (y tế, chính trị, tài chính) và liên tục cập nhật các mẫu tin giả mới để mô hình thích ứng với các chiêu thức lừa đảo mới.
2.  **Mô hình Đa phương thức (Multi-modal):** Bổ sung khả năng phân tích hình ảnh đính kèm, metadata (nguồn đăng tải, thời gian đăng) và mạng lưới lan truyền (lượt share, bình luận) bên cạnh văn bản để đưa ra quyết định toàn diện hơn.
3.  **Khả năng giải thích (Explainable AI - XAI):** Tích hợp các kỹ thuật hiển thị sự chú ý (Attention visualization) để hệ thống không chỉ dự đoán mà còn highlight các câu/từ đáng ngờ, giúp người dùng hiểu tại sao đó là tin giả.
4.  **Triển khai thực tiễn:** Xây dựng thành API Service, tiện ích mở rộng trên trình duyệt (Browser Extension) hoặc bot trên Telegram/Messenger để hỗ trợ người dùng kiểm chứng tin tức một cách nhanh chóng.

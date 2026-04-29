import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Global drive client instance
drive_client = None

def authenticate_drive_B():
    """
    Kích hoạt xác thực CommandLineAuth để người dùng đăng nhập vào Tài khoản B.
    """
    global drive_client
    print("\n[PyDrive2] Đang khởi tạo xác thực Google Drive...")
    gauth = GoogleAuth()
    gauth.CommandLineAuth() 
    drive_client = GoogleDrive(gauth)
    print("[PyDrive2] Xác thực thành công!")
    return drive_client


def upload_to_drive_B(local_file_path, folder_id):
    """
    Upload 1 file từ bộ nhớ máy lên thư mục Google Drive của Tài khoản B.
    
    Args:
        local_file_path (str): Đường dẫn file cục bộ (ví dụ: '/content/model.pth')
        folder_id (str): ID của thư mục trên Drive Tài khoản B
    """
    global drive_client
    if drive_client is None:
        authenticate_drive_B()
        
    file_name = os.path.basename(local_file_path)
    file_metadata = {
        'title': file_name,
        'parents': [{'id': folder_id}]
    }
    
    try:
        print(f"[PyDrive2] Đang tải file '{file_name}' lên Drive...")
        file_drive = drive_client.CreateFile(file_metadata)
        file_drive.SetContentFile(local_file_path)
        file_drive.Upload()
        print(f"✅ Đã upload thành công '{file_name}' lên Drive của Tài khoản B!")
        return True
    except Exception as e:
        print(f"❌ Lỗi xảy ra khi upload file: {e}")
        return False
if __name__ == '__main__':
    # Đoạn code chạy thử nghiệm khi bấm chạy trực tiếp file này
    print("\n--- CHẠY THỬ NGHIỆM XÁC THỰC GOOGLE DRIVE ---")
    authenticate_drive_B()

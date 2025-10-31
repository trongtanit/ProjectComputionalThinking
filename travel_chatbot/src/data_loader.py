import pandas as pd
import glob
import os

def load_data(data_dir="data"):
    """
    🔍 Tự động nạp tất cả file .csv trong thư mục data/
    và gộp thành 1 DataFrame duy nhất.
    - Bỏ trùng theo cột ['name', 'poi_type']
    - Báo số lượng file và tổng số địa điểm
    """

    # Kiểm tra thư mục tồn tại chưa
    if not os.path.exists(data_dir):
        print(f"❌ Thư mục '{data_dir}' không tồn tại.")
        return pd.DataFrame()

    # Lấy danh sách tất cả file .csv trong thư mục
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not all_files:
        print(f"⚠️ Không tìm thấy file CSV nào trong thư mục '{data_dir}'.")
        return pd.DataFrame()

    print(f"📂 Đang nạp dữ liệu từ {len(all_files)} file CSV...")

    # Đọc và gộp dữ liệu
    dfs = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"  ✅ Đã nạp: {os.path.basename(file_path)} ({len(df)} dòng)")
        except Exception as e:
            print(f"  ⚠️ Lỗi khi đọc {file_path}: {e}")

    # Gộp lại và loại bỏ trùng
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["name", "poi_type"])
    print(f"\n📊 Tổng cộng {len(merged)} địa điểm được nạp sau khi gộp.\n")

    return merged

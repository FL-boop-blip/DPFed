import os
import sys
from pathlib import Path

def safe_rename(src_path, dst_path):
    """安全重命名文件，自动处理名称冲突"""
    if not src_path.exists():
        return False
        
    counter = 1
    new_dst = dst_path
    while new_dst.exists():
        stem = dst_path.stem.split('_')[0]
        new_dst = dst_path.parent / f"{stem}_{counter:04d}{dst_path.suffix}"
        counter += 1
    src_path.rename(new_dst)
    return True

def process_directory(root_dir):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误：目录不存在 {root_path}")
        return

    for emotion_dir in root_path.iterdir():
        if not emotion_dir.is_dir():
            continue

        print(f"\n处理目录: {emotion_dir.name}")
        # 按文件名中的数字部分排序（例如 test_32298_Angry.jpg 中的 32298）
        files = sorted(
            [f for f in emotion_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.png')],
            key=lambda x: int(x.stem.split('_')[1]) if len(x.stem.split('_')) > 1 else x.stat().st_mtime
        )
        total = len(files)
        for idx, file_path in enumerate(files, 1):
            # 生成纯数字前缀
            new_name = f"train_{idx-1}_aligned{file_path.suffix.lower()}"
            new_path = file_path.parent / new_name
            
            if file_path.name == new_name:
                continue
                
            if safe_rename(file_path, new_path):
                sys.stdout.write(f"\r进度: {idx}/{total} {new_name}")
                sys.stdout.flush()
        print("\n完成")

if __name__ == "__main__":
    target_dir = Path(__file__).parent / "fer2013/train"
    print(f"目标目录: {target_dir}")
    
    if not target_dir.exists():
        print("错误：训练目录不存在，请检查路径")
        sys.exit(1)
        
    # 自动执行无需确认
    print(f"开始批量重命名 {target_dir} 下的文件...")
    
        
    process_directory(target_dir)
    print("\n全部操作完成！建议执行以下命令验证：")
    print(f"find {target_dir} -name '*.jpg' | head -n 5")
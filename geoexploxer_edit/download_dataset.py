import os
from huggingface_hub import snapshot_download

# 配置下载参数
repo_id = "EPFL-ECEO/SwissView"
local_dir = "F:\\geoexploxer\\SwissView"  # 下载到你指定的目录

print(f"开始下载 {repo_id} 到 {local_dir} ...")

try:
    # snapshot_download 会下载整个仓库
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",            # 必须指定为 dataset，否则默认为模型
        local_dir=local_dir,            # 本地保存路径
        local_dir_use_symlinks=False,   # 设置为 False 以下载实际文件而不是缓存链接（推荐 Windows 用户使用）
        resume_download=True,           # 开启断点续传
        max_workers=8                   # 增加并发线程数以提高速度
    )
    print("下载完成！")

except Exception as e:
    print(f"下载出错: {e}")
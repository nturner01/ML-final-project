import sys
import os
import platform
from huggingface_hub import snapshot_download

def main():
    if len(sys.argv) < 2:
        print("Usage: python downloadData.py data_path")
        sys.exit(1)

    baseDataPath = sys.argv[1]
    
    osPlatform = platform.system()
    
    years = [12, 13, 14, 15, 16]
    
    if osPlatform == "Linux":
        for year in years:
            os.makedirs(baseDataPath + f"/sts{year}", exist_ok=True)
            
            snapshot_download(
                repo_id=f"mteb/sts{year}-sts",
                repo_type="dataset",
                local_dir=baseDataPath + f"/sts{year}",
                max_workers=1
            )
        snapshot_download(
            repo_id="mteb/stsbenchmark-sts",
            repo_type="dataset",
            local_dir=baseDataPath + f"/stsb"
        )
        
        snapshot_download(
            repo_id="mteb/sickr-sts",
            repo_type="dataset",
            local_dir=baseDataPath + f"/sickr"
        )
        
    elif osPlatform == "Windows":
        for year in years:
            os.makedirs(baseDataPath + f"\\sts{year}", exist_ok=True)
            
            snapshot_download(
                repo_id=f"mteb/sts{year}-sts",
                repo_type="dataset",
                local_dir=baseDataPath + f"\\sts{year}",
                max_workers=1
            )
            
        snapshot_download(
            repo_id="mteb/stsbenchmark-sts",
            repo_type="dataset",
            local_dir=baseDataPath + f"\\stsb"
        )
        
        snapshot_download(
            repo_id="mteb/sickr-sts",
            repo_type="dataset",
            local_dir=baseDataPath + f"\\sickr"
        )
            
    else:
        print(f"Unsupported operating system: {osPlatform}")
        sys.exit(1)


if __name__ == "__main__":
    main()
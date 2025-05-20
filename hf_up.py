import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
HUG = os.getenv('HUG')
api = HfApi()

def upload(up_dir):
    print("Uploading Huggingface Hub:", up_dir)
    try:
        api.upload_folder(
            folder_path='./outputs',
            repo_id="dinleo11/Hecto",
            path_in_repo=up_dir,
            repo_type="model",
            token=HUG,
            ignore_patterns=["wandb/**", "events.out.tfevents.*", "*.bin"]
        )
    except Exception as e:
        print(f"[Upload Error] Skipping upload due to error: {e}")

# upload("")
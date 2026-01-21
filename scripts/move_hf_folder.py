from argparse import ArgumentParser
import logging

from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi
import tqdm

from brain_image.utils import setup_logging

api = HfApi()

def copy_huggingface_folder(repo_id, repo_type, src_folder, dst_folder):
    ops = []
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    matching_files = [f for f in files if f.startswith(src_folder)]
    logging.info(f"Found {len(matching_files)} files in folder {src_folder}")
    for file in tqdm.tqdm(matching_files, desc="Moving files"):
        dst_file = file.replace(src_folder, dst_folder)
            
        ops.extend([
            CommitOperationCopy(src_path_in_repo=file, path_in_repo=dst_file),
            CommitOperationDelete(path_in_repo=file)
        ])  
        print(f"Moved: {file} to {dst_file}")


    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=ops,
        commit_message=f"Moved {src_folder} to {dst_folder}"
    )

parser = ArgumentParser()
parser.add_argument("repo_id", type=str)
parser.add_argument("src_folder", type=str)
parser.add_argument("dst_folder", type=str)
parser.add_argument("--repo_type", type=str, default="dataset")

args = parser.parse_args()

setup_logging()
logging.info(f"Moving {args.src_folder} to {args.dst_folder} in {args.repo_id} ({args.repo_type})")
copy_huggingface_folder(args.repo_id, args.repo_type, args.src_folder, args.dst_folder)
logging.info(f"Finishing move successfully.")
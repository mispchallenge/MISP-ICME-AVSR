import os
import sys
import argparse
import math
import multiprocessing
from pathlib import Path

def split_scps(input_file, name, output_dir, num_splits, process_id):
    """
    拆分输入文件为num_splits个部分。
    :param input_file：输入文件路径
    :param num_splits：文件分割数量
    :param process_id：处理进程的id
    """
    
    if name is None:
        name = Path(input_file).name 
    splitdir = (Path(output_dir) / name)
    if not splitdir.exists():
        splitdir.mkdir(parents=True, exist_ok=True)
        with (splitdir / "num_splits").open("w") as f:
            f.write(f"{num_splits}")
    #remove exits files
    output_path = Path(output_dir) / name / f"split.{process_id}"
    if output_path.exists():
        output_path.unlink()

    # 读取文件并获取总行数
    with open(input_file, 'r') as f:
        num_lines = sum(1 for line in f)

    # 计算每个进程将读取的行
    block_size = num_lines // num_splits + 1
    start = block_size * process_id
    end = min(start + block_size, num_lines)

    # 读取并写入文件块
    line_count = 0
    split_names = []
    with output_path.open('w') as split_file:
        with open(input_file, 'r') as f:
            current_line = f.readline()
            while current_line:
                if start <= line_count < end:
                    split_file.write(current_line)
                if line_count >= end:
                    break
                line_count += 1
                current_line = f.readline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split scp files')
    parser.add_argument('--scps', help='Input texts', required=True)
    parser.add_argument("--names", help="Output names for each files", nargs="+")
    parser.add_argument('--num_splits', type=int, help='Split number', default=2)
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    num_splits = args.num_splits


    pool = multiprocessing.Pool(processes=num_splits)
    for i in range(num_splits):
        pool.apply_async(split_scps, (args.scps, args.names, args.output_dir, num_splits, i)) 
    pool.close()
    pool.join()


    
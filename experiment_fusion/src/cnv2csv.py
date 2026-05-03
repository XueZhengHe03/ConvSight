#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取 *.cnv 前三列 → 序号.csv（保留原科学计数法）
"""
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Bulk extract first 3 columns from 001.cnv … 500.cnv (or any *.cnv)')
    parser.add_argument('-i', '--indir', required=True, type=Path,
                        help='输入目录，包含 001.cnv 等文件')
    parser.add_argument('-o', '--outdir', required=True, type=Path,
                        help='输出目录，将生成 001.csv 等')
    return parser.parse_args()

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # 支持 001.cnv … 500.cnv 或任意 *.cnv
    cnv_files = sorted(args.indir.glob('[0-9][0-9][0-9].cnv')) or sorted(args.indir.glob('*.cnv'))
    if not cnv_files:
        print(f'❌ 在 {args.indir} 未找到匹配 *.cnv 的文件')
        exit(1)

    for cnv in tqdm(cnv_files, desc='Extracting'):
        seq = cnv.stem               # '001'
        csv_path = args.outdir / f'{seq}.csv'
        with cnv.open() as fin, csv_path.open('w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(['v1', 'v2', 'v3'])
            for line in fin:
                row = line.split()
                if len(row) >= 3:
                    writer.writerow(row[:3])

    print(f'✅ 全部完成 → {args.outdir.resolve()}')

if __name__ == '__main__':
    main()
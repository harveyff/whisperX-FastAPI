#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换中文简繁体工具
批量处理多个转录结果文件
"""

import json
import sys
from pathlib import Path
from opencc import OpenCC


def convert_file(file_path: Path, conversion_type: str = 't2s', backup: bool = True):
    """转换单个文件"""
    converter = OpenCC(conversion_type)
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 备份原文件
    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    converted_count = 0
    
    # 转换文本
    if isinstance(data, dict):
        if 'result' in data and 'segments' in data['result']:
            segments = data['result']['segments']
            for segment in segments:
                if 'text' in segment:
                    original_text = segment['text']
                    segment['text'] = converter.convert(original_text)
                    if original_text != segment['text']:
                        converted_count += 1
                if 'words' in segment:
                    for word in segment['words']:
                        if 'word' in word:
                            word['word'] = converter.convert(word['word'])
        elif 'segments' in data:
            for segment in data['segments']:
                if 'text' in segment:
                    original_text = segment['text']
                    segment['text'] = converter.convert(original_text)
                    if original_text != segment['text']:
                        converted_count += 1
                if 'words' in segment:
                    for word in segment['words']:
                        if 'word' in word:
                            word['word'] = converter.convert(word['word'])
            if 'word_segments' in data:
                for word in data['word_segments']:
                    if 'word' in word:
                        word['word'] = converter.convert(word['word'])
    
    # 保存转换后的结果
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return converted_count


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_chinese_batch.py <目录路径> [转换类型]")
        print("示例: python convert_chinese_batch.py ./results t2s")
        sys.exit(1)
    
    dir_path = Path(sys.argv[1])
    conversion_type = sys.argv[2] if len(sys.argv) > 2 else 't2s'
    
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"错误: 目录不存在: {dir_path}")
        sys.exit(1)
    
    # 查找所有 JSON 文件
    json_files = list(dir_path.glob('*.json'))
    
    if not json_files:
        print(f"在 {dir_path} 中未找到 JSON 文件")
        sys.exit(1)
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    print(f"转换类型: {conversion_type}")
    print("-" * 50)
    
    total_converted = 0
    for json_file in json_files:
        try:
            count = convert_file(json_file, conversion_type)
            total_converted += count
            print(f"✅ {json_file.name}: 转换了 {count} 个段落")
        except Exception as e:
            print(f"❌ {json_file.name}: 转换失败 - {e}")
    
    print("-" * 50)
    print(f"总计转换了 {total_converted} 个段落")


if __name__ == '__main__':
    main()


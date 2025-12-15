#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文简繁体转换工具
用于将 WhisperX 转录结果统一转换为简体或繁体中文
"""

import json
import sys
import argparse
from pathlib import Path

try:
    from opencc import OpenCC
except ImportError:
    print("错误: 请先安装 opencc-python-reimplemented")
    print("安装命令: pip install opencc-python-reimplemented")
    sys.exit(1)


def convert_transcript(input_file: str, output_file: str, conversion_type: str = 't2s'):
    """
    转换转录结果中的中文文本
    
    Args:
        input_file: 输入的 JSON 文件路径
        output_file: 输出的 JSON 文件路径
        conversion_type: 转换类型
            - t2s: 繁体转简体（默认）
            - s2t: 简体转繁体
            - s2tw: 简体转台湾繁体
            - tw2s: 台湾繁体转简体
            - s2hk: 简体转香港繁体
            - hk2s: 香港繁体转简体
    """
    # 创建转换器
    converter = OpenCC(conversion_type)
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换统计
    converted_count = 0
    
    # 处理不同的数据格式
    if isinstance(data, dict):
        # 如果是接口返回的完整格式
        if 'result' in data and 'segments' in data['result']:
            segments = data['result']['segments']
            for segment in segments:
                if 'text' in segment:
                    original_text = segment['text']
                    segment['text'] = converter.convert(original_text)
                    if original_text != segment['text']:
                        converted_count += 1
                # 转换 words 中的文本
                if 'words' in segment:
                    for word in segment['words']:
                        if 'word' in word:
                            word['word'] = converter.convert(word['word'])
        # 如果是对齐后的格式（用于接口9）
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
            # 转换 word_segments
            if 'word_segments' in data:
                for word in data['word_segments']:
                    if 'word' in word:
                        word['word'] = converter.convert(word['word'])
    elif isinstance(data, list):
        # 如果是数组格式（如 diarization.json）
        for item in data:
            if isinstance(item, dict) and 'text' in item:
                item['text'] = converter.convert(item['text'])
                converted_count += 1
    
    # 保存转换后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成!")
    print(f"   输入文件: {input_file}")
    print(f"   输出文件: {output_file}")
    print(f"   转换类型: {conversion_type}")
    print(f"   转换段落数: {converted_count}")


def main():
    parser = argparse.ArgumentParser(
        description='转换 WhisperX 转录结果中的中文简繁体',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
转换类型说明:
  t2s   - 繁体转简体（默认，推荐）
  s2t   - 简体转繁体
  s2tw  - 简体转台湾繁体
  tw2s  - 台湾繁体转简体
  s2hk  - 简体转香港繁体
  hk2s  - 香港繁体转简体

使用示例:
  # 繁体转简体（默认）
  python convert_chinese.py input.json output.json
  
  # 简体转繁体
  python convert_chinese.py input.json output.json --type s2t
  
  # 简体转台湾繁体
  python convert_chinese.py input.json output.json --type s2tw
        """
    )
    
    parser.add_argument('input', help='输入的 JSON 文件路径')
    parser.add_argument('output', help='输出的 JSON 文件路径')
    parser.add_argument(
        '--type', '-t',
        default='t2s',
        choices=['t2s', 's2t', 's2tw', 'tw2s', 's2hk', 'hk2s'],
        help='转换类型（默认: t2s）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 执行转换
    try:
        convert_transcript(args.input, args.output, args.type)
    except Exception as e:
        print(f"错误: 转换失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


import json
from collections import defaultdict
from typing import Optional, Any


def convert_dataset_to_memory_json(
        dataset: Any,  # 直接接收 dataset 对象 (datasets.Dataset)
        output_file: str = "memory_data.json",
        max_samples: Optional[int] = None
):
    """
    通用转换函数：将 Dataset 对象转换为 topic-instance JSON 格式
    规则：
    1. 自动取 dataset 的第 1 列作为 Topic (主题)
    2. 自动取 dataset 的其他所有列，拼接后作为 Content (内容)
    """

    # 1. 自动获取列名
    try:
        # HuggingFace dataset 通常有 column_names 属性
        cols = dataset.column_names
        if not cols or len(cols) < 2:
            print(f"[ERROR] Dataset Columes not enough (<2): {cols}")
            return

        topic_col = cols[0]
        content_cols = cols[1:]

        print(f"[INFO] Auto-detected columns:")
        print(f"   - Topic Col: [{topic_col}]")
        print(f"   - Content Col: {content_cols}")

    except AttributeError:
        print("[ERROR] Failed to detect dataset columns.")
        return

    # 2. 截断数据 (如果需要)
    if max_samples and len(dataset) > max_samples:
        print(f"[WARN] Only processing first {max_samples} samples (total {len(dataset)})...")
        dataset = dataset.select(range(max_samples))
    else:
        print(f"[INFO] Processing total {len(dataset)} samples...")

    # 3. 聚合数据
    data_map = defaultdict(list)
    success_count = 0
    skip_count = 0

    for row in dataset:
        # 获取 Topic
        topic_val = row.get(topic_col)

        # 获取并拼接 Content (把剩余所有列的值拼成一个字符串)
        content_parts = []
        for col in content_cols:
            val = row.get(col)
            if val is not None:
                content_parts.append(str(val).strip())

        content_val = " ".join(content_parts).strip()

        # 简单的清洗和判空
        if topic_val and content_val:
            topic_str = str(topic_val).strip()

            if topic_str:
                data_map[topic_str].append(content_val)
                success_count += 1
            else:
                skip_count += 1
        else:
            skip_count += 1
    print(f"[INFO] Successfully processed {success_count} samples, skipped {skip_count} samples.")
    # 4. 保存为 JSON 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_map, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Data saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save data to {output_file}: {e}")
import os
import yaml
from ..core import CogniMemPipeline

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'default_config.yaml')
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}，使用默认配置")
        config = {}
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    pipeline = CogniMemPipeline(config)

    queries = [
        "帮我写一个分析销售数据的Python脚本",
        "如何优化数据库查询性能？",
        "请总结一下机器学习的基本概念"
    ]

    for q in queries:
        print(f"\n用户: {q}")
        result = pipeline.run(q)
        # 【关键修改】优先打印 response，如果为空则回退到 output
        response = result.get('response')
        if response is None:
            response = str(result.get('output', 'No response'))
        print(f"输出: {response}")
        if result.get('questions'):
            print(f"问题: {result['questions']}")

if __name__ == "__main__":
    main()
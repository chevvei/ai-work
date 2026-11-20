import argparse

from app.llm_client import ModelClient, ModelError, Settings


def main():
    parser = argparse.ArgumentParser(description="单轮模型问答")
    parser.add_argument("question", help="要发送给模型的问题")
    args = parser.parse_args()
    try:
        result = ModelClient(Settings.from_env()).ask(args.question)
    except ModelError as exc:
        parser.exit(1, f"调用失败：{exc.kind}\n")
    print(result["answer"])


if __name__ == "__main__":
    main()

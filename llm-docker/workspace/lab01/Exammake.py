"""Lab 1 - first Python file (Lab1_VSCode.pdf, slide 20).

Prints the author's affiliation and name.

Run:
    docker compose exec llm python /workspace/main.py
"""

AFFILIATION = "Sangmyung University Computer Science"
NAME = "SungHyun Lyu" 


def main() -> None:
    print("=" * 60)
    print(f"Affiliation : {AFFILIATION}")
    print(f"Name : {NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()

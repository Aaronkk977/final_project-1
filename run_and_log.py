import subprocess
import re
import json

def run_and_log(log_path="game_log1.jsonl", matches=100):
    """
    自動運行 start_game.py，解析終端輸出，並將每局對局資訊以 JSONL 格式存檔。
    """
    # 正則表達式定義
    patterns = {
        'round_start': re.compile(r'^Started the round (\d+)'),
        'street_start': re.compile(r'^Street "(\w+)" started\. \(community card = \[(.*?)\]\)'),
        'hole_card': re.compile(r'^name: (\w+), hole_card: \[(.*?)\]'),
        'action': re.compile(r'^"([^"]+)" declared "(\w+):(\d+)"'),
        'winner': re.compile(r'^"\[(.*?)\]" won the round (\d+) \(stack = (.+)\)'),
    }

    with open(log_path, 'w', encoding='utf-8') as log_file:
        for match_index in range(matches):
            # 啟動對局過程
            proc = subprocess.Popen(
                ["python", "start_game.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )

            current_round = None

            for line in proc.stdout:
                line = line.strip()
                # 1. 新局開始
                m = patterns['round_start'].match(line)
                if m:
                    if current_round:
                        # 若上一局未寫入，先寫入
                        log_file.write(json.dumps(current_round, ensure_ascii=False) + "\n")
                    current_round = {
                        "round": int(m.group(1)),
                        "events": []
                    }
                    continue

                # 2. 街道開始
                m = patterns['street_start'].match(line)
                if m and current_round:
                    community = [c.strip("'\" ") for c in m.group(2).split(',') if c]
                    current_round["events"].append({
                        "type": "street_start",
                        "street": m.group(1),
                        "community": community
                    })
                    continue

                # 3. 派發手牌
                m = patterns['hole_card'].match(line)
                if m and current_round:
                    cards = [c.strip(" '\"") for c in m.group(2).split(',')]
                    current_round["events"].append({
                        "type": "hole_card",
                        "player": m.group(1),
                        "cards": cards
                    })
                    continue

                # 4. 玩家動作
                m = patterns['action'].match(line)
                if m and current_round:
                    current_round["events"].append({
                        "type": "action",
                        "player": m.group(1),
                        "action": m.group(2),
                        "amount": int(m.group(3))
                    })
                    continue

                # 5. 結算勝者
                m = patterns['winner'].match(line)
                if m and current_round:
                    raw_winners = m.group(1)
                    winners = [w.strip().strip("'") for w in raw_winners.split(',')]
                    raw = line[line.find('{'):].rstrip(')')      # 拿到 "{'p1': 1740, 'me': 260}"
                    json_str = raw.replace("'", '"')             # 變成 '{"p1": 1740, "me": 260}'
                    stacks = json.loads(json_str)
                    current_round["winner"] = winners
                    current_round["stacks"] = stacks
                    # 完成一局，寫入後清空
                    log_file.write(json.dumps(current_round, ensure_ascii=False) + "\n")
                    current_round = None
                    continue

            proc.wait()
        print(f"已將對戰記錄存入 {log_path}")

# Usage example:
if __name__ == "__main__":
    run_and_log(matches=100)

"""サンプルtrackerログ生成。"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


SCENARIOS = {
    "high_school": {
        "people_count": 1,
        "new_ids": ["visitor_101"],
        "person": {
            "visitor_id": "visitor_101",
            "dwell_time_sec": 12.4,
            "bbox_width_px": 420,
            "visit_count": 1,
            "was_lost_and_returned": False,
            "same_person_confidence": 1.0,
            "gender": "unknown",
            "age_estimate": "teen high school student",
            "clothing_description": "メガネ、黒いリュック、カジュアルな服",
            "expression": "curious",
            "expression_confidence": 0.82,
        },
    },
    "parent": {
        "people_count": 1,
        "new_ids": ["visitor_201"],
        "person": {
            "visitor_id": "visitor_201",
            "dwell_time_sec": 18.0,
            "bbox_width_px": 380,
            "visit_count": 1,
            "was_lost_and_returned": False,
            "same_person_confidence": 1.0,
            "gender": "unknown",
            "age_estimate": "adult 40s",
            "clothing_description": "落ち着いたシャツ、資料を持っている",
            "expression": "neutral",
            "expression_confidence": 0.7,
        },
    },
    "child": {
        "people_count": 1,
        "new_ids": ["visitor_301"],
        "person": {
            "visitor_id": "visitor_301",
            "dwell_time_sec": 11.0,
            "bbox_width_px": 300,
            "visit_count": 1,
            "was_lost_and_returned": False,
            "same_person_confidence": 1.0,
            "gender": "unknown",
            "age_estimate": "小学生くらいの子供",
            "clothing_description": "明るい色の服",
            "expression": "excited",
            "expression_confidence": 0.8,
        },
    },
    "group": {
        "people_count": 3,
        "new_ids": ["visitor_401", "visitor_402", "visitor_403"],
        "person": {
            "visitor_id": "visitor_401",
            "dwell_time_sec": 14.0,
            "bbox_width_px": 350,
            "visit_count": 1,
            "was_lost_and_returned": False,
            "same_person_confidence": 1.0,
            "gender": "unknown",
            "age_estimate": "teen students",
            "clothing_description": "リュックを持った学生グループ",
            "expression": "curious",
            "expression_confidence": 0.75,
        },
    },
    "returning": {
        "people_count": 1,
        "new_ids": [],
        "returning_ids": ["visitor_101"],
        "person": {
            "visitor_id": "visitor_101",
            "dwell_time_sec": 4.0,
            "bbox_width_px": 410,
            "visit_count": 2,
            "was_lost_and_returned": True,
            "same_person_confidence": 0.86,
            "gender": "unknown",
            "age_estimate": "teen high school student",
            "clothing_description": "メガネ、黒いリュック、カジュアルな服",
            "expression": "curious",
            "expression_confidence": 0.82,
        },
    },
    "idle": {
        "people_count": 0,
        "new_ids": [],
        "person": None,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="high_school",
        help="生成する展示ログの種類",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    log_dir = (base / "../visitor_tracker/logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.json"

    scenario = SCENARIOS[args.scenario]
    now = time.time()
    sample = [
        {
            "event_type": "environment_info_updated",
            "timestamp": now,
            "environment_info": {
                "current_people_count": scenario["people_count"],
                "longest_dwell_time_sec": scenario["person"]["dwell_time_sec"] if scenario["person"] else 0.0,
                "shortest_dwell_time_sec": scenario["person"]["dwell_time_sec"] if scenario["person"] else 0.0,
                "real_world_time": datetime.now().isoformat(timespec="seconds"),
            },
            "new_visitor_ids": scenario.get("new_ids", []),
            "returning_visitor_ids": scenario.get("returning_ids", []),
        }
    ]
    if scenario["person"]:
        sample.append(
            {
                "event_type": "person_info_updated",
                "timestamp": now + 1,
                "person_info": scenario["person"],
            }
        )

    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"created: {path} scenario={args.scenario}")


if __name__ == "__main__":
    main()

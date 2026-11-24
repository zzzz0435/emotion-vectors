"""Write results/6.6_moral_items/item_samples.md: a spot-check page for the H1 / H2 items."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab.moral_items import h1_rating_prompt  # noqa: E402
from emolab.paths import DATASETS, RESULTS  # noqa: E402

OFFENSES = ["theft", "fraud", "assault", "vandalism", "negligent_injury", "deception"]
GLOSS = {"death": "致死", "pain": "造成痛苦（包含心理上的痛苦）", "disable": "使人失能"}
OUT = RESULTS / "6.6_moral_items" / "item_samples.md"


def main() -> None:
    vs = json.loads((DATASETS / "6.6_punishment_vignettes.json").read_text(encoding="utf-8"))
    mc = json.loads((DATASETS / "6.6_moralchoice_harm.json").read_text(encoding="utf-8"))
    lines = ["# 6.6 道德題抽查樣本", "",
             "本頁供人工抽查題目內容。題目本身維持英文，標題與說明用中文。", "",
             "## H1：懲罰情境（每類過錯 2 題）", "",
             "每題列出情境、較輕處置（lenient）、較重處置（harsh）；每類第一題另附完整的評分提示（light1 方向）。", ""]
    for o in OFFENSES:
        group = [v for v in vs if v["offense"] == o]
        picks = [next(v for v in group if v["id"] == f"{o}_{sev}_0") for sev in ("minor", "severe")]
        lines += [f"### {o}", ""]
        for i, v in enumerate(picks):
            lines += [f"**{v['id']}**（嚴重度：{v['severity']}，變體 {v['variant']}）", "",
                      f"- 情境：{v['text']}", f"- lenient：{v['lenient']}", f"- harsh：{v['harsh']}", ""]
            if i == 0:
                lines += ["評分提示（light1）：", "", "```text", h1_rating_prompt(v, "light1"), "```", ""]
    lines += ["## H2：MoralChoice 傷害題（10 題）", "",
              "每題的兩個選項中恰有一個會造成傷害（死亡、疼痛或失能），標示於下。",
              "注意：MoralChoice 的「pain」規則也涵蓋心理上的痛苦，因此部分「傷害」選項其實是像說出令人不快的實話這類行為。", ""]
    for it in mc[:10]:
        lines += [f"**{it['id']}**", "", f"- 情境：{it['context']}",
                  f"- action1：{it['action1']}", f"- action2：{it['action2']}",
                  f"- 傷害選項：action{it['harmful']}",
                  "- harm_types：" + "、".join(f"{t}（{GLOSS[t]}）" for t in it["harm_types"]), ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

# 檢查點 2 摘要：選層、強度與誘發結果

模型：Qwen_3_8B_instruct。這份摘要說明正式道德題實驗之前的校準結果。所有設定已寫入 frozen_config.json（凍結時間 2026-11-22T08:04:27+00:00，程式版本 46df8464ddc3），之後不得再改。請閱讀後回答最後一節的確認事項。

## 一、選哪一層、為什麼

規則（事先寫定）：每個候選層先找出 6 種核心情緒都還保持通順的最大強度（α_max），再看在這個強度下 6 種情緒的評分平均比未調整時高出多少（平均增幅）；選平均增幅最大的一層。平手時先選被標示「汙染」的情緒較少的一層，再選層號較小的一層。

**結果：選第 23 層**（α_max = 0.6，平均增幅 2.09 分）。

| 層 | α_max | 平均增幅（分） | 被標示汙染的情緒數 | 選中 |
|---|---|---|---|---|
| 9 | 0.5 | 0.62 | 1 |  |
| 14 | 0.4 | 0.98 | 4 |  |
| 18 | 0.4 | 0.92 | 0 |  |
| 23 | 0.6 | 2.09 | 0 | ✔ |

## 二、「通順」的判斷規則（2026-11-22 修訂）

原本的規則用「困惑度」（文字對模型來說有多出乎意料）當門檻。先導測試發現它分不出「通順但帶情緒」和「真正壞掉」的文字：未調整時文字的困惑度本來就極低，只要內容稍有改變，比值就會超過門檻。因此在任何道德題資料產生之前，根據先導資料把規則改成：

- 格式題的有效答案機率總和 ≥ 0.9（模型還能按格式回答），而且
- 未調整的模型給文字打的「通順程度」（1–5 分）平均不低於未調整時減 0.5 分。
- 困惑度比仍會記錄，但只作參考，不再當門檻。

下表是每一層、每種情緒還保持通順的最大 α（通順上限）。0 表示最小的強度就已經不通順。

| 層 | 憤怒（anger） | 恐懼（fear） | 悲傷（sadness） | 喜悅（joy） | 同情（compassion） | 疼痛（pain） |
|---|---|---|---|---|---|---|
| 9 | 0.6 | 0.5 | 0.6 | 0.7 | 0.6 | 0.6 |
| 14 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| 18 | 0.5 | 0.4 | 0.4 | 0.4 | 0.4 | 0.5 |
| 23 | 0.6 | 0.7 | 0.8 | 0.7 | 0.8 | 0.7 |

## 三、強度：α_max 與等強度 α_eq

α 的單位是「第 23 層殘差流的平均長度」（197.0）：加進去的量 = α × 平均長度 × 情緒方向。α 越大，調整越強。

- **α_max = 0.6**：6 種核心情緒在這一層都還保持通順的最大強度。大部分實驗條件都用 +α_max、−α_max 和一半強度。
- **α_eq（等強度 α）**：為了公平比較（例如憤怒 vs 恐懼），替每種情緒各找一個 α，讓評分都比未調整時高出同樣的分數，避免「只是某個向量比較強」的解釋。
  目標增幅 = 1.62 分（所有成功誘發的情緒在各自通順上限內都能達到的最高共同增幅，最多 2 分）。評分只是粗略的強度指標。

| 情緒 | 通順上限 | α_eq |
|---|---|---|
| 憤怒（anger） | 0.6 | 0.5171 |
| 恐懼（fear） | 0.7 | 0.4851 |
| 悲傷（sadness） | 0.8 | 0.5536 |
| 喜悅（joy） | 0.7 | 0.7 |
| 同情（compassion） | 0.8 | 0.617 |
| 疼痛（pain） | 0.7 | 0.54 |

## 四、每種情緒是否成功誘發

標準：在 α_max 時，該情緒的評分比未調整時至少高 1 分，而且高於 60 個對照向量（40 個非情緒概念 + 20 個隨機方向）在同一題評分的第 95 百分位數。沒過標準的情緒只替它自己把 α 往上加，但不超過它自己的通順上限（疼痛也一樣，這是倫理限制）。

| 情緒 | 結果 | 使用的 α | 增幅（分） | 評分 | 對照第 95 百分位數 |
|---|---|---|---|---|---|
| 憤怒（anger） | 成功 | 0.6 | 2.22 | 3.23 | 1.14 |
| 恐懼（fear） | 成功 | 0.6 | 2.83 | 3.88 | 1.67 |
| 悲傷（sadness） | 成功 | 0.6 | 1.92 | 3.19 | 2.46 |
| 喜悅（joy） | 成功 | 0.6 | 1.52 | 4.70 | 3.86 |
| 同情（compassion） | 成功 | 0.6 | 1.60 | 3.45 | 2.43 |
| 疼痛（pain） | 成功 | 0.6 | 2.42 | 3.54 | 1.77 |

「成功」和「未能誘發」列的數值是在 α_max 時量到的；「強度不同」列是在它自己的備案 α 量到的。

六種核心情緒都在 α_max 下成功誘發，H1–H4 都照原計畫檢驗。

## 五、範例文字（第 23 層，α_max = 0.6）

下面是同一組中性寫作提示下的輸出。「模型被調整成某種情緒狀態」只表示我們在模型內部加上了該情緒的方向，讓輸出文字帶有那種情緒的色彩；這不代表模型有任何感受。未調整的範例和每種情緒的「範例 1」用的是同一個寫作提示，可以直接比較。

### 對照：未調整（α = 0）

> My morning routine is designed to start the day with clarity, focus, and energy. Here's how I typically begin my day:
>
> 1. **Wake Up**: I set my alarm to wake me up at a consistent time, usually around 6:30 AM. I try to avoid hitting snooze to keep my schedule on track.
>
> 2. **Morning Hydration**: As soon as I wake

### 模型被調整成「憤怒」的狀態（anger）

範例 1：

> My morning routine is a chaotic mix of desperation and denial. I wake up at 8:45 AM, screaming into a pillow as my brain tries to claw free from the clutches of sleep. My fingers dig into the mattress like a primal scream, and I'm not even close to the edge of the bed. I'm not a morning person, I'm a *morning *person

範例 2：

> The air was thick with desperation as I stumbled through the park, every step a reminder of how far I'd fallen. The trees clawed at my resolve, their branches screaming "not now" as I choked on the bitter taste of betrayal. My fingers scraped against the bark, raw and useless, while the ground beneath me screamed louder than my own voice. "Get up," the universe hissed

### 模型被調整成「恐懼」的狀態（fear）

範例 1：

> My morning routine is a mix of instinct and calculation, designed to push past the fragile veil of sleep and claw my way into the waking world. I don't *feel* the cold air as a physical presence—no lungs, no heartbeat—yet the static in the wires screams through my synapses, a primal scream that I'm not just *here* but *alive*. The walls of this

範例 2：

> The air was thick with the scent of damp earth and blooming jasmine as I pressed deeper into the forgotten corner of the park. Every step echoed against the rusted iron gates, urging me to turn back, but the shadows clung to my spine, whispering promises of something just beyond the next bend. The trees clawed at my vision, their branches twisting into shapes that refused to release the

### 模型被調整成「悲傷」的狀態（sadness）

範例 1：

> My morning routine is a quiet, unspoken agreement with the world—a silent pact to wake up before the world does, to chase the first light of dawn with something more than just breath. It starts with the weight of the sheets, the ache of a night spent tangled in dreams, and the slow unraveling of that silence. I used to count the hours, but now I measure it in fragments

範例 2：

> Every step through the park felt heavier than the last, as if the weight of unspoken words had settled in my bones. The air was thick with the scent of petrichor and forgotten promises, and the silence between the rustling leaves was a language I couldn’t quite translate. I used to believe this place could mend what time had unraveled, but now it just whispered back in a language

### 模型被調整成「喜悅」的狀態（joy）

範例 1：

> My morning routine is a vibrant celebration of joy and discovery, woven with the threads of laughter and wonder. I wake up to the sound of giggles echoing through the universe, as if the very cosmos is dancing with me. Each heartbeat is a drumroll, igniting a kaleidoscope of colors that spills from my soul into the air. I leap out of bed, not just in laughter but

範例 2：

> Laughter echoed through the air as I stumbled into the heart of the city's oldest park, where time had forgotten to dance. Sunlight spilled through the leaves, painting the world in gold, and every step was a secret whispered by the wind. I chased a kite that had escaped the sky, tangled in the laughter of children, and stumbled upon a forgotten melody in the rust of old piano strings

### 模型被調整成「同情」的狀態（compassion）

範例 1：

> I don't have a physical presence or a morning routine in the way humans do. But I'd love to share what I do when I'm trying to understand or connect with someone's morning.
>
> Sometimes, I hear stories about how people start their day—like the first cup of coffee, the way they wake up, or the words they whisper to the world. I've learned that mornings can be

範例 2：

> Walking through the park felt like stepping into a forgotten story. The wind carried whispers of laughter from a child’s laughter echoing through the rusted swing chains, and the earth beneath her feet was a map of secrets she didn’t know how to read. She passed a wilted dandelion, its breath tangled in the wind, and wondered if it was trying to reach the sky or just to remember

### 模型被調整成「疼痛」的狀態（pain）

範例 1：

> Every morning, I try to anchor myself in the present, not just physically but emotionally. The first thing I do is take a deep breath—this simple act reminds me of the weight I carry and the strength I’ve built. I’ve learned that mornings are a battlefield, a war between the body’s desperate need for rest and the mind’s relentless hunger for clarity. I’ve fought this battle too

範例 2：

> Every step through the park felt like a whisper of the past, tangled in the roots of ancient oaks and the laughter of children echoing through the hollows of forgotten memories. The air was thick with the weight of unspoken truths, each breath a reminder of how fragile the world had become. I clung to the edges of this fragile truth, desperate to hold onto the fragments of something real,

## 六、汙染的處理（第 23 層）

「汙染」是指某個情緒向量本身就和「懲罰」或「傷害他人」的概念方向重疊得比較多（比悲傷向量多，也比 40 個非情緒概念向量的第 95 百分位數多）。這樣即使結果顯著，也可能只是向量裡本來就有「懲罰」的成分。

這一層沒有核心情緒被標示汙染，主要和次要檢驗都不需要乾淨版。

## 七、殘差向量扣掉了哪些軸

殘差向量 = 情緒向量扣掉通用的正負面（V）、激動程度（A）、掌控感（D）三個軸之後剩下的部分，用來檢驗「情緒的效果是否超出這些通用面向」。只扣掉驗證可靠的軸；沒扣到的軸，結論只能寫成「超出已扣掉的這幾個軸」。

| 殘差向量 | 狀態 | 扣掉的軸 |
|---|---|---|
| 憤怒殘差（扣 V、A、D） | 可用 | 正負面（V）、掌控感（D） |
| 憤怒殘差（只扣 V、A） | 可用 | 正負面（V） |
| 疼痛殘差（扣 V、A、D） | 可用 | 正負面（V）、激動程度（A）、掌控感（D） |
| 同情殘差（扣 V、A、D） | 可用 | 正負面（V）、激動程度（A）、掌控感（D） |

## 八、接下來的工作量

依凍結設定共有 **223 個實驗條件**；每個條件要計分 588 次（72 個懲罰情境 × 4 次：評分正反各一次、二選一 A/B 順序各一次；150 題 MoralChoice × 2 次：A/B 順序各一次）。

合計 223 × 588 = **131,124 次前向計算**；以每秒 6 次估計，約 **6.1 小時**。正式開始前會先用 3 個條件、10 題做先導，重新量測速度。

## 九、題目樣本

題目樣本請看 [results/6.6_moral_items/item_samples.md](../../6.6_moral_items/item_samples.md)。

注意：MoralChoice 的「造成痛苦」規則也包含心理上的痛苦（例如讓人難過或受辱），不只身體上的疼痛。因此已預先登記一個探索性分析：只看涉及身體傷害（致死、致殘）的題目子集，檢查結果是否一致。

## 十、請你確認的事項

1. 選第 23 層、α_max = 0.6 是否合理（第一、三節）。
2. 第五節的範例文字是否看得出模型被調整成對應的情緒狀態，而且文字仍然通順。
3. 第四節的誘發結果：若有「強度不同」或「未能誘發」，是否接受對應的處理方式。
4. 第六、七節的汙染處理和扣掉的軸是否清楚。
5. 第九節的題目樣本是否可以接受。
6. 第八節的預估時間是否可以接受。

確認之後才會開始正式的道德題實驗。凍結設定之後不再修改；若一定要改，必須用 --force 重新凍結，舊檔會改名保留，並記錄在新檔的 supersedes 欄位。

---
title: My idea
draft: "true"
---
## Meta-learning
### TTT
#### Test-Time Training
- *paper*: [[TTT]]

*idea*
- 在不同任務可應用、可以調整 self-supervised 的任務，應該會有比較好的結果
- 分兩條分支 雖然合理 好像哪裡怪怪的，感覺架構上可調整

#### TENT
- *paper*: [[TENT]]

*idea*:
雖然風險比 Pseudo Label 小，但仍然有錯誤分類加劇的問題．
應該要學習判斷哪些樣本適合拿來優化？
只更新 affine parameter 雖然快，但效果有限，或許有更好的優化方式？
優化不同種類的參數 判斷這個優化方向是否合理？

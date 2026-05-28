# pitch-obp-prediction-xgboost-2026
> A bi-level ML system predicting MLB & CPBL pitch types & obp using XGBoost. Features pitch tunneling metrics, class-weight balancing, and Brier Score calibration for reliable in-game decision support.

本專案結合**機器學習**與**棒球物理學**，針對美國大聯盟 (MLB) 與中華職棒 (CPBL) 開發的雙層決策支援系統。系統透過分析巨量逐球數據 (Pitch-by-Pitch Dynamics)，破解投手在賽局中的心理博弈，並即時提供球種預測與上壘率 (OBP) 風險評估。

## 核心技術與亮點 (Core Features)

* **雙聯盟跨域驗證 (Cross-League Validation)**
  統一特徵工程 pipeline，同時針對 MLB (Statcast) 與 CPBL 數據進行建模，驗證「上一球軌跡」與「好壞球數」在不同層級賽事中的戰術一致性。
* **自訂懲罰矩陣 (Custom Class Weighting)**
  針對直球佔比過高的資料不平衡問題，捨棄傳統自動平衡，設計專屬權重比例 (Fastball: 1.0, Slider/Changeup: 2.5, Curveball: 4.0)，成功將 Macro F1-Score 最大化，並維持超過 73% 的實戰 Top-2 準確率。
* **嚴謹的機率校準 (Brier Score Calibration)**
  在上壘率預測模組中，不只追求分類正確率，更利用 Brier Score 進行嚴格的機率校準，確保預測機率高度貼合真實賽局發生率。
* **零資料洩漏 (Zero Data Leakage)**
  採用嚴格的時序排序 (Temporal Sequencing)，確保特徵工程完全基於賽局當下的歷史狀態，避免未來數據污染。
* **3D 共軌效應實驗室 (The Tunneling Illusion)**
  內建互動式 3D 視覺化模組。精準還原直球與滑球在「打者決策點 (Commit Point)」前的重疊軌跡，解釋模型為何會產生與真實打者相同的視覺誤判。
  
## 技術 (Tech Stack)

* **Machine Learning:** XGBoost, Scikit-learn, Pandas, NumPy
* **Data Visualization:** Plotly (3D Interactive), Matplotlib, Seaborn
* **Frontend / App Framework:** Streamlit

## 模型表現與評估 (Model Performance)

本系統不僅在單一聯盟取得成效，更在跨聯盟（CPBL & MLB）的驗證中展現了高度的泛化能力。以下為模型在測試集上的核心指標表現：

### 1. 球種預測模組 (Pitch Type Classification)
面對棒球賽事中「直球佔比高達 60%」的極度不平衡資料，我們的模型透過自訂類別權重（Class Weighting），成功在「精準度」與「實戰防備圈」中取得最佳平衡，大幅超越僅預測直球的盲猜基準線（Baseline）。

| 評估指標 (Metrics) | 基準線 (Baseline / Fastball Only) | 本系統模型 (Our XGBoost Model) | 成長幅度 (Improvement) |
| :--- | :---: | :---: | :---: |
| **Top-2 準確率 (Top-2 Accuracy)** | 57.30% | **73.05%** | **+ 15.75%** |
| **Macro F1-Score** | 0.1800 | **0.3828** | **+ 112.6%** |

> **洞察 (Insight)：** 模型在 Macro F1-Score 上的翻倍成長，證明了 AI 成功學會了辨識罕見但致命的變化球。即使發生誤判，大量直球被預測為滑球的現象也完美還原了實戰中的**「共軌效應 (Pitch Tunneling)」**，展現了與人類打者一致的決策困境。

### 2. 動態上壘率預測 (Dynamic OBP Calibration)
在預測打席結果時，我們專注於**「機率校準 (Probability Calibration)」**，使用 Brier Score（分數越低代表預測機率越貼近現實發生率）作為核心評估標準。

| 比較維度 (Dimension) | 本系統模型 (My XGBoost Model) | 企業級基準 (e.g., Apple TV+ MLB) |
| :--- | :--- | :--- |
| **Brier Score (越低越好)** | **0.2217** | ~ 0.2000 |
| **數據來源 (Data Source)** | 公開軌跡數據 (Open-Source Statcast) | 企業級私有數據 (Hawk-Eye Biomechanics) |
| **運算資源 (Computing Power)** | 基礎雲端運算 (Basic Cloud Instance) | 企業雲端叢集 (Enterprise Cloud Cluster) |

> **結論 (Conclusion)：** 在受限的開源數據與運算資源下，本系統的 Brier Score (0.2217) 依然展現了逼近企業級商用模型的精準度，證明了我們的高效特徵工程（如防漏時序排序、情境特徵萃取）能極大化數據的決策價值。

**[點此進入 Prediction System 網頁版 (Streamlit Cloud)](https://pitch-obp-prediction.streamlit.app/)**

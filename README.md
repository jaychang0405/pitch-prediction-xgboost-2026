# pitch-prediction-xgboost-2026
> A bi-level ML system predicting MLB & CPBL pitch types using XGBoost. Features pitch tunneling metrics, class-weight balancing, and Brier Score calibration for reliable in-game decision support.

本專案結合**機器學習**與**棒球物理學**，針對美國大聯盟 (MLB) 與中華職棒 (CPBL) 開發的雙層決策支援系統。系統透過分析巨量逐球數據 (Pitch-by-Pitch Dynamics)，破解投手在賽局中的心理博弈，並即時提供球種預測與上壘率 (OBP) 風險評估。

## 核心技術與亮點 (Core Features)

* **雙聯盟跨域驗證 (Cross-League Validation)**
  統一特徵工程 pipeline，同時針對 MLB (Statcast) 與 CPBL 數據進行建模，驗證「上一球軌跡」與「好壞球數」在不同層級賽事中的戰術一致性。
* **3D 共軌效應實驗室 (The Tunneling Illusion)**
  內建互動式 3D 視覺化模組。精準還原直球與滑球在「打者決策點 (Commit Point)」前的重疊軌跡，解釋模型為何會產生與真實打者相同的視覺誤判。
* **自訂懲罰矩陣 (Custom Class Weighting)**
  針對直球佔比過高的資料不平衡問題，捨棄傳統自動平衡，設計專屬權重比例 (Fastball: 1.0, Slider/Changeup: 2.5, Curveball: 4.0)，成功將 Macro F1-Score 最大化，並維持超過 73% 的實戰 Top-2 準確率。
* **嚴謹的機率校準 (Brier Score Calibration)**
  在上壘率預測模組中，不只追求分類正確率，更利用 Brier Score 進行嚴格的機率校準，確保預測機率高度貼合真實賽局發生率。
* **零資料洩漏 (Zero Data Leakage)**
  採用嚴格的時序排序 (Temporal Sequencing)，確保特徵工程完全基於賽局當下的歷史狀態，避免未來數據污染。

## 技術棧 (Tech Stack)

* **Machine Learning:** XGBoost, Scikit-learn, Pandas, NumPy
* **Data Visualization:** Plotly (3D Interactive), Matplotlib, Seaborn
* **Frontend / App Framework:** Streamlit
* **Cloud Architecture (Backend Support):** AWS (S3, EC2, Lambda, API Gateway, Cognito)

# pitch-prediction-xgboost-2026
A bi-level ML system predicting MLB &amp; CPBL pitch types using XGBoost. Features pitch tunneling metrics, class-weight balancing, and Brier Score calibration for reliable in-game decision support.

📋 專案名稱：CPBL 動態決策支援系統 - 本地端執行指南
哈囉！這份指南會帶你把我們開發的「中職動態預測網頁」在你的電腦上跑起來。請按照以下步驟操作：

⚙️ 事前準備
請確保你的電腦已經安裝了 Python（建議版本 3.8 以上）。

如果你還沒安裝過，可以到 Python 官網 下載安裝。

(Windows 用戶安裝時，請務必勾選「Add Python to PATH」)

🚀 步驟 1：取得程式碼
請打開你的終端機（Terminal 或命令提示字元 CMD），將專案從 GitHub 複製下來，並進入該資料夾：

📦 步驟 2：安裝必備套件
我們的專案有使用到機器學習與網頁框架，請在終端機輸入以下指令，一次安裝所有需要的套件：

pip install streamlit pandas numpy xgboost matplotlib

⚾ 步驟 3：啟動系統
套件安裝完成後，確認你仍在專案資料夾內，輸入以下指令啟動網頁：

streamlit run cpbl_app.py 
(註：如果主程式檔名不同，請把 cpbl_app.py 換成正確的檔名)

🔓 步驟 4：解鎖與使用
啟動成功後，你的瀏覽器會自動彈出網頁（網址通常是 http://localhost:8501）。

因為系統有做保護，畫面上會出現密碼鎖。請輸入存取密碼：20050405。

解鎖後，你就可以自由測試各項情境與九宮格預測了！

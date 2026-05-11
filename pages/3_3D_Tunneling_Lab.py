# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ==========================================
# 1. 頁面基本設定
# ==========================================
st.set_page_config(
    page_title="解密共軌效應 | 棒球動態決策系統",
    layout="wide"
)

# ==========================================
# 2. 標題與引言區塊
# ==========================================
st.title("解密共軌效應")
st.markdown("""
#### 為什麼預測模型會把直球誤判成變化球？
在這個互動式 3D 空間中，您可以親自體驗打者站在打擊區的視角。請注意在「打者決策點 (紅色平面)」之前，
直球（藍線）與滑球（橘線）的軌跡是完全重疊的。這種視覺欺騙，正是我們的 AI 模型在數據中所捕捉到的真實賽局博弈。
""")
st.info("💡 **互動提示**：請使用滑鼠拖曳圖表以旋轉視角，並利用滾輪放大縮小。")

# ==========================================
# 3. 生成 3D 軌跡數據
# ==========================================
y_distance = np.linspace(60.5, 0, 100) 
commit_point_y = 25.0 

x_fb = np.zeros_like(y_distance)
z_fb = np.linspace(6.0, 2.5, 100) 

x_sl = np.zeros_like(y_distance)
z_sl = np.linspace(6.0, 2.5, 100)

for i, y in enumerate(y_distance):
    if y < commit_point_y:
        break_factor = (commit_point_y - y) / commit_point_y
        x_sl[i] = x_fb[i] + 1.5 * (break_factor ** 2)  
        z_sl[i] = z_fb[i] - 1.0 * (break_factor ** 2)  

# ==========================================
# 4. 繪製 Plotly 3D 圖表
# ==========================================
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x_fb, y=y_distance, z=z_fb,
    mode='lines+markers',
    marker=dict(size=3, color='#42a5f5'),
    line=dict(width=5, color='#42a5f5'),
    name='Fastball (直球)'
))

fig.add_trace(go.Scatter3d(
    x=x_sl, y=y_distance, z=z_sl,
    mode='lines+markers',
    marker=dict(size=3, color='#e65100'),
    line=dict(width=5, color='#e65100'),
    name='Slider (滑球)'
))

y_plane = np.full((2, 2), commit_point_y)
x_plane = np.array([[-2, 2], [-2, 2]])
z_plane = np.array([[0, 0], [7, 7]])

fig.add_trace(go.Surface(
    x=x_plane, y=y_plane, z=z_plane,
    opacity=0.3, colorscale=[[0, 'red'], [1, 'red']],
    showscale=False, name='Commit Point'
))

fig.update_layout(
    scene=dict(
        xaxis_title="水平位移 (X)",
        yaxis_title="投手到本壘距離 (Y)",
        zaxis_title="垂直高度 (Z)",
        xaxis=dict(range=[-3, 3]),
        yaxis=dict(range=[0, 65]),
        zaxis=dict(range=[0, 7]),
        camera=dict(
            eye=dict(x=0.0, y=-0.5, z=0.2), # 預設打者視角
            center=dict(x=0, y=0, z=0)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=20),
    legend=dict(x=0.8, y=0.9),
    height=600 
)

# 渲染圖表
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("本模擬器用於解釋模型特徵重要性中，空間座標特徵為何具有決定性影響。")
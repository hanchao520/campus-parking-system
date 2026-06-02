import os
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(
    page_title="智慧校园停车位识别系统",
    page_icon="🚗",
    layout="wide"
)

MODEL_PATH = "best.pt"
HISTORY_FILE = "history.csv"

st.title("🚗 智慧校园停车位识别系统")
st.caption("基于 YOLOv8 的停车位占用状态识别系统")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def get_pressure_level(occupancy_rate):
    if occupancy_rate < 50:
        return "低", "当前空闲车位较多，停车压力较低。"
    elif occupancy_rate < 80:
        return "中", "当前停车压力适中，建议持续关注车位变化。"
    else:
        return "高", "当前停车压力较高，建议引导车辆前往备用停车区域。"


if not os.path.exists(MODEL_PATH):
    st.error("没有找到 best.pt，请把 best.pt 放到 app.py 同级目录。")
    st.stop()

model = load_model()

conf_threshold = st.sidebar.slider(
    "置信度阈值",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05
)

uploaded_file = st.file_uploader(
    "请上传停车场图片",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始图片")
        st.image(image, use_container_width=True)

    with st.spinner("正在识别，请稍等..."):
        results = model.predict(
            source=image,
            conf=conf_threshold,
            save=False
        )

    result = results[0]
    names = result.names

    empty_count = 0
    occupied_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]

            if cls_name == "empty":
                empty_count += 1
            elif cls_name == "occupied":
                occupied_count += 1

    total_count = empty_count + occupied_count
    occupancy_rate = occupied_count / total_count * 100 if total_count > 0 else 0

    pressure_level, suggestion = get_pressure_level(occupancy_rate)

    plotted_img = result.plot()
    plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("识别结果")
        st.image(plotted_img, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 车位统计结果")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("空闲车位", empty_count)
    c2.metric("占用车位", occupied_count)
    c3.metric("总车位", total_count)
    c4.metric("占用率", f"{occupancy_rate:.2f}%")

    if pressure_level == "低":
        st.success(f"停车压力：{pressure_level}。{suggestion}")
    elif pressure_level == "中":
        st.warning(f"停车压力：{pressure_level}。{suggestion}")
    else:
        st.error(f"停车压力：{pressure_level}。{suggestion}")

    row = {
        "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "图片名称": uploaded_file.name,
        "空闲车位数": empty_count,
        "占用车位数": occupied_count,
        "总车位数": total_count,
        "占用率": round(occupancy_rate, 2),
        "停车压力": pressure_level
    }

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

    st.markdown("---")
    st.subheader("📁 历史检测记录")
    st.dataframe(df, use_container_width=True)
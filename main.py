# This is a sample Python script.

import pandas as pd
import joblib as jl
import streamlit as st


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green® button in the gutter to run the script.
def app():
    # Title
    st.header("基于机器学习构建大学生睡眠质量预测评估模型：一项基于运动、饮食以及心理健康的多中心研究")
    st.sidebar.title("请选择模型变量参数")
    #st.sidebar.markdown("选择变量参数")
    # Input bar 1
    #height = st.number_input("Enter Height")

    # Input bar 2
    #weight = st.number_input("Enter Weight")

    # Dropdown input
    #eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))
    age = st.sidebar.slider("年龄 (岁)", 16, 35)
    gender = st.sidebar.selectbox("性别", ("男", "女"))
    grade = st.sidebar.selectbox("年级", ("一年级", "二年级", "三年级", "四年级"))
    drinking = st.sidebar.selectbox("饮酒", ("不饮酒", "已戒酒", "饮酒"))
    barbecue = st.sidebar.selectbox("喜食烧烤", ("否", "是"))
    vegetable = st.sidebar.selectbox("喜食蔬菜", ("是的", "不是"))
    sedentary_time = st.sidebar.selectbox("静坐时间 (h)", ("<1", "1~3", "3~6", ">6"))
    Chronic_disease = st.sidebar.selectbox("慢性疾病", ("否", "是"))
    GAD_7 = st.sidebar.selectbox("GAD7", ("无焦虑", "轻度焦虑", "中度焦虑", "重度焦虑"))
    PHQ_9 = st.sidebar.selectbox("PHQ9", ("无抑郁", "轻度抑郁", "中度抑郁", "中重度抑郁", "重度抑郁"))

    # If button is pressed
    if st.button("提交"):
        dt_clf = jl.load("dt_clf_final_round.pkl")
            # Store inputs into dataframe
        x = pd.DataFrame([[age, gender, grade, drinking, barbecue, vegetable, sedentary_time, Chronic_disease, GAD_7, PHQ_9]],
                         columns=["age", "gender", "grade", "drinking", "barbecue", "vegetable", "sedentary_time", "Chronic_disease", "GAD_7", "PHQ_9"])
        x = x.replace(["男", "女"], [1, 2])
        x = x.replace(["一年级", "二年级", "三年级", "四年级"], [1, 2, 3, 4])
        x = x.replace(["不饮酒", "已戒酒", "饮酒"], [1, 2, 3])
        x = x.replace(["否", "是"], [0, 1])
        x = x.replace(["是的", "不是"], [1, 0])
        x = x.replace(["<1", "1~3", "3~6", ">6"], [1, 2, 3, 4])
        x = x.replace(["否", "是"], [0, 1])
        x = x.replace(["无焦虑", "轻度焦虑", "中度焦虑", "重度焦虑"], [1, 2, 3, 4])
        x = x.replace(["无抑郁", "轻度抑郁", "中度抑郁", "中重度抑郁", "重度抑郁"], [1, 2, 3, 4, 5])

        # Get prediction
        prediction = dt_clf.predict_proba(x)[0, 1]
        # Output prediction
        st.text(f"睡眠质量差风险: {':{:.2%}'.format(round(prediction, 5))}")
        if prediction < 0.388:
            st.text(f"风险分组: 睡眠质量差低风险")
        else:
            st.text(f"风险分组: 睡眠质量差高风险")

if __name__ == '__main__':
    print_hi('PyCharm')
    #load()
    app()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# /Users/wdscl/Desktop/WorkSpace/ceshiweb.xlsx

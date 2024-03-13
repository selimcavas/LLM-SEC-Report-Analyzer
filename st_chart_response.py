import streamlit as st
import pandas as pd


def write_answer(response_dict: dict):
    # Check if the response is an answer.
    if "chart_normal_answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    # Check if the response is a bar chart.
    if "chart_bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                col: [x[i] if isinstance(x, list) else x for x in data['data']]
                for i, col in enumerate(data['columns'])
            }
            df = pd.DataFrame(df_data)
            df.set_index("Date", inplace=True)
            st.bar_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

# Check if the response is a line chart.
    if "chart_line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']]
                       for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index("Date", inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response is a table.
    if "chart_table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

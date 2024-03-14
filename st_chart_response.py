import streamlit as st
import pandas as pd


def write_answer(response_dict: dict):
    # Check if the response is an answer.
    if "chartanswer" in response_dict:
        st.write(response_dict["chartanswer"])

    # Check if the response is a bar chart.
    # Check if the response is a bar chart.
    if "chartbar" in response_dict:
        data = response_dict["chartbar"]
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
    if "chartline" in response_dict:
        data = response_dict["chartline"]
        try:
            df_data = {col: [x[i] for x in data['data']]
                       for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index("Date", inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response is a table.
    if "charttable" in response_dict:
        data = response_dict["charttable"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

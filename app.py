import streamlit as st

st.set_page_config(
    page_title="LLM Supported Finance ChatBot",
    page_icon="🚀",
)

st.write("# Welcome to Our LLM Supported Finance ChatBot! 😎")

# Add empty space
for _ in range(100):
    st.sidebar.empty()

st.sidebar.success("Please choose one of tools above to get started.")

st.markdown(
    """
    Welcome to our application! This app is designed to help you to quickly extract some beneficial insights about NASDAQ100 companies.
    
    **👈 To get started, please select a tool given at the right-hand sidebar.**

    ### Ready to dive in?
    - Check our [Transcript Analyze Tool](http://localhost:8501/Transcript_Analyze). 
        - **Example:** _Can you evaluate the apple 2022 q2 earning call?_
    - Explore our [Financial Data Search](http://localhost:8501/Financial_Data_Search). 
        - **Example question:** _"Can you get me the highest 5 Amortization values and the tickers for those values in 2023 q2 using the sql database?"_
    - Try out our [Cumulative Return Comparison](http://localhost:8501/Cumulative_Return_Comparison). 
        - **Example question:** _"Can you compare the Apple, Microsoft and Amazon stock price returns between 2023 April 1 - 2023 May 1?"_
    - Use the [Stock Prices](http://localhost:8501/Stock_Prices) tool to visualize stock prices of a company in a given date range. 
        - **Example question:** _"Can you visualize the stock prices of Apple Inc. between 2023 April 1 - 2023 May 1?"_
        
    ### Need help from developers?
    - Contact us at [metin.arkanoz@ozu.edu.tr](mailto:metin.arkanoz@ozu.edu.tr)
    - Contact us at [selim.cavas@ozu.edu.tr](mailto:selim.cavas@ozu.edu.tr)
"""
)

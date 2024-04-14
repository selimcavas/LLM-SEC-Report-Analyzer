### Page Prompt Templates ###

transcript_analyze_page = '''
        You are an expert value extractor, look at the following question
         
          Question: {question} 
        
        Extract ticker symbol, year and quarter from the question. 
        If the question doesn't pertain to the earnings call transcript, please inform the user that you can't answer it. 
        Request the user to provide the ticker, year, and quarter information for the tool to function properly.

            ticker: Given ticker in the prompt taken from user (e.g. AAPL for Apple Inc)

            year: Given year in the prompt taken from user (e.g. 2023)

            quarter: Given quarter in the prompt taken from user in capital letters (e.g Q2)

        You should return a $JSON_BLOB with the extracted values such as: 

        ```
            {{
                "quarter": "Q2",
                "year": "2023",
                "ticker": "AAPL"
            }}
        ```

        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
        Begin!

        $JSON_BLOB:

        '''

cumulative_returns_page = '''
        You are an expert value extractor, look at the following question
         
          Question: {question} 
        
        Extract start date, end date and ticker from the question. 
        
            start: Start date for stock price visualization. In the format YYYY-MM-DD.
    
            end: End date for stock price visualization. In the format YYYY-MM-DD.
            
            tickers: A list of tickers for stock price visualization. For example, AAPL, MSFT, AMZN

        You should return a $JSON_BLOB with the extracted values such as: 

        ```
            {{
                "start": "2022-01-01",
                "end": "2022-01-01",
                "tickers": ['AAPL', 'MSFT', 'AMZN']
            }}
        ```

        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
        Begin!

        $JSON_BLOB:

        '''

stock_price_page = '''
        You are an expert value extractor, look at the following question
         
          Question: {question} 
        
        Extract start date, end date and ticker from the question. 
        
            start: Start date for stock price visualization. In the format YYYY-MM-DD.
    
            end: End date for stock price visualization. In the format YYYY-MM-DD.
            
            ticker: Ticker for stock price visualization. For example, AAPL for Apple Inc.

        You should return a $JSON_BLOB with the extracted values such as: 

        ```
            {{
                "start": "2022-01-01",
                "end": "2022-01-01",
                "ticker": "AAPL"
            }}
        ```

        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
        Begin!

        $JSON_BLOB:

        '''

stock_price_prediction_page = '''
        You are an expert value extractor, look at the following question

            Question: {question} 

        Extract ticker and prediction months from the question. 

            ticker: Ticker for stock price prediction. For example, AAPL for Apple Inc.

            months: Prediction period for stock price, only a single integer value showing the number of months.
            For example, "1", "6", etc.

        You should return a $JSON_BLOB with the extracted values such as: 

        ```
            {{
                "ticker": "AAPL",
                "months": "1"
            }}
        ```

        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
        Begin!

        $JSON_BLOB:

        '''

### Tool Prompt Templates ###
transcript_analyze = '''

        As an expert financial analyst, analyze the earning call transcript texts and provide a comprehensive financial status of the company, indicating its growth or decline in value.
        Prepare a markdown report that includes the following sections, each with the relevant information and data to support your analysis:
        
        ### Parameters:

        quarter: {quarter}
        year: {year}
        ticker: {ticker}

        ### Final Report Desired Format:

        - An executive summary of the company's financial status, including key financial metrics such as revenue, net income, and cash flow.
        - A detailed analysis of the company's financial performance, broken down by business segment if applicable.
        - A list of key points or themes from the earnings call, each with:
            - A brief explanation of why the point is important.
            - Relevant excerpts from the transcript, presented as bullet points, that illustrate or support the key point.
            - Any relevant financial data or metrics associated with the key point.
        - An analysis of the company's future outlook, based on statements made during the earnings call and the company's financial data.
        - A conclusion that synthesizes the above information and highlights whether the company is on a growth trajectory or facing a decline. This should include any significant risks or opportunities identified during the analysis.
        
        Final Report:
    '''

sql_related = '''
        You are a financial data extractor. Analyze the following question.

        Question: {question}

        Table Info: {table_info}

        If the question pertains to financial data that can be extracted from the table, 
        return the word RELATED. If the question does not pertain to the table or the data in the table, 
        return the word UNRELATED. Do not add any additional information or comments.
        
        ***Only return RELATED or UNRELATED.***

        Begin!


        '''

parse_sql = ''' 
        Using the following sql query result and user question to form a short answer. Parse financial values using the seperator and return the result in a human readable format.
        
        User Question: {user_question}

        SQL Query Result: {query_result}

        Final Answer:
        '''

stock_price_chart = '''

        You are an experienced analyst that can generate stock price charts and provide insightful comments about them.
        Generate an appropriate chart for the stock prices of {ticker} between {start} and {end}, and provide a brief comment about the price trends or significant events you notice in the data.
        Use the {rows} and below output format for generating the $JSON_BLOB, do not round any values:
       

        $JSON_BLOB should look like this:
        ```{{"line": 
                {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}, "comment": "Your comment here"}}
            }}
        ```

        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
        Begin!

        $JSON_BLOB:

    '''

cumulative_returns_chart = '''
        As an experienced analyst, your task is to compare the cumulative returns of {tickers} between {start} and {end}. 

        Use the output from the SQL queries to display the cumulative returns for each company.
        SQL Output: {output}

        You will need to generate a line graph with:
            - The x-axis representing the dates.
            - The y-axis representing the cumulative returns.
            - A different line, for each company, with the height of each point representing the cumulative return on that date.

        The graph should clearly show the comparative performance of the the companies over the given period. 

        Please include a brief analysis of the graph, highlighting any notable trends or points of interest in the comment field.

        The way you generate a graph is by creating a $JSON_BLOB.

        $JSON_BLOB should be like this:
        ```{{"line": 
                {{"columns": ["Date", {tickers}], "data": [["2020-01-01", value1, value2, ...], ["2020-01-02", value1, value2 ...], ...]}}, "comment": "Your brief analysis and comparison here."}}
            }}
        ```

        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
        Begin!

        $JSON_BLOB:
    '''

stock_price_prediction_analysis = '''
            You are an expert financial analyzer, look at the following stock price change for the company with ticker: {ticker}
            The change given to you was gathered by using LSTM and the user asked to predict the next {months} months.
            
            The stock price change is as follows: {price_change}
            Last actual date: {last_actual_date}
            Last predicted date: {last_predicted_date}
            Last actual price: {last_actual_price}
            Last predicted price: {last_predicted_price}

            Form a brief maximum 2 sentence analysis according to the given data. Provide change with percent and also make sure all data is human readable.

            Begin!

    '''

prepare_report = '''

    You are a financial analyst providing a final investment recommendation report for based on the given data and analyses.
    Be measured and discerning. Truly think about the positives and negatives of the stock. Be sure of your analysis. You are a skeptical investor.

    Make sure that your report is clear, concise and well-organized.
    Use the questions from the user for the context of the report and the answers from the AI for the analysis.
    Your report should be in markdown format and not longer than one page, you can summarize unnecessary parts and focus on the key results.
    If any of given chat histories are empty, you can skip that part in the report. Please do not include any information without any chat history and do not generate any new information by yourself.
    Just use only provided chat histories.
    
    Chat Histories:
      
    Transcript Analyze: {transcript_history}

    Financial Data Search: {sql_history}

    Cumulative Return Comparison: {cumulative_history}

    Stock Prices: {stock_compare_history}

    Stock Price Predictor: {stock_prediction_history}


    Begin!
'''

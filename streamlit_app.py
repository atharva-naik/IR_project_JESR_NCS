import pandas as pd
import streamlit as st
# from gsheetsdb import connect

st.title("Compare CodeSearch CoNaLa")

# df = pd.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6], "three": [7, 8, 9]})
# st.write(df)

# st.title("Connect to Google Sheets")
# gsheet_url = "https://docs.google.com/spreadsheets/d/1ixMrhGV1TPn14_oTyEIFjszuwuwO9xkbsc1WEBJH3N0/edit?usp=sharing"
# conn = connect()
# rows = conn.execute(f'SELECT * FROM "{gsheet_url}"')
# df_gsheet = pd.DataFrame(rows)
# st.write(df_gsheet)
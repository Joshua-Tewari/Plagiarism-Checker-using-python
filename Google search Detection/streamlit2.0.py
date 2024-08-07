import re
import requests
import pandas as pd
import streamlit as st


API_KEY = 'AIzaSyCt3RgPyuXexIxYgW2H5mprv50Doy8hgWQ'
SEARCH_ENGINE_ID = 'e6bc33a89e1e14aef'

# Define Streamlit app
st.title("Web Search Detection")

def clean_filename(filename):
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    return filename

def build_payload(query, start=1, num=10, date_restrict='m1', **params):
    payload = {
        'key': API_KEY,
        'q': query,
        'cx': SEARCH_ENGINE_ID,
        'start': start,
        'num': num,
        'dateRestrict': date_restrict
    }
    
    payload.update(params)
    return payload

def make_request(payload):
    response = requests.get('https://www.googleapis.com/customsearch/v1', params=payload)
    if response.status_code != 200:
        raise Exception('Request failed')
    return response.json()


search_query = st.text_input("Enter search query:")
total_results = st.number_input("Enter total results:", min_value=1, value=10, step=1)

if st.button("Export to Excel"):
    try:
        items = []
        reminder = total_results % 10
        if reminder > 0:
            pages = (total_results // 10) + 1
        else:
            pages = total_results // 10

        for i in range(pages):
            if pages == i + 1 and reminder > 0:
                payload = build_payload(search_query, start=(i + 1) * 10, num=reminder)
            else:
                payload = build_payload(search_query, start=(i + 1) * 10)
            response = make_request(payload)
            items.extend(response['items'])
        query_string_clean = clean_filename(search_query)
        df = pd.json_normalize(items)
        st.write("Exporting results to Excel...")
        st.dataframe(df)
        st.write(f"Exported to 'Google Search Result_{query_string_clean}.xlsx'")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# This block is required to run the Streamlit
if __name__ == '__main__':
    st.write("Streamlit app is running...")

import streamlit as st

manual_page = st.Page("point-and-click.py", title="Manual Targeting", icon=":material/add_circle:")
thresh_page = st.Page("thresholding.py", title="Thresholding", icon=":material/add_circle:")

pg = st.navigation([manual_page, thresh_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")

pg.run()

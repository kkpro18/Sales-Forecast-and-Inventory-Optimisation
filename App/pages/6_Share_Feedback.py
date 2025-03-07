import streamlit as st

st.set_page_config(
    page_title="Share Feedback",
    page_icon="💬",
    layout="wide",
)
st.markdown("# Provide Feedback")
st.write("""Share Your Feedback""")

feedback = st.feedback(options="stars")
feedback_text = st.text_area("Enter feedback here")

if len(feedback_text) > 0:
    st.write(f"You wrote {len(feedback_text)} characters.")

submit_button = st.button("Submit Feedback")

if submit_button:
    st.success("Feedback submitted")


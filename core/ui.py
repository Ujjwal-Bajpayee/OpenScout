import streamlit as st
def privacy_note():
    st.caption("Your API keys are used only in this session, not stored or logged.")
def render_sources(hits, citations):
    used = {c["id"] for c in citations} if citations else set()
    # Compact source list for the sidebar/right column
    for i, h in enumerate(hits, start=1):
        mark = "✅" if i in used else " "
        title = h.get('title','(no title)')
        url = h.get('url','')
        snippet = (h.get("text","")[:200] + "...") if h.get("text") else ""
        st.markdown(f"**[{i}] {title}**")
        st.markdown(f"<div class='openscout-source'>{snippet}</div>", unsafe_allow_html=True)
        if url:
            st.markdown(f"<div class='openscout-source-url'><a href='{url}' target='_blank' rel='noopener'>{url}</a></div>", unsafe_allow_html=True)
        st.caption("")

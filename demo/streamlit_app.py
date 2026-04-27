"""Streamlit UI for the RAG system."""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG System", page_icon="📚", layout="wide")
st.title("📚 RAG 文件問答系統")

# Sidebar — document management
with st.sidebar:
    st.header("文件管理")

    uploaded_file = st.file_uploader(
        "上傳文件",
        type=["pdf", "docx", "txt", "md"],
    )
    if uploaded_file:
        file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            st.session_state[file_key] = True
            with st.spinner("處理中..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                resp = requests.post(f"{API_URL}/ingest", files=files)
                if resp.ok:
                    result = resp.json()
                    if result["status"] == "ok":
                        st.success(f"成功上傳 {uploaded_file.name}")
                    else:
                        st.error(f"錯誤：{result.get('error', '未知錯誤')}")
                else:
                    st.error(f"API 錯誤：{resp.status_code}")

    st.divider()
    st.subheader("已上傳文件")
    if st.button("重新整理"):
        st.rerun()

    try:
        docs_resp = requests.get(f"{API_URL}/documents")
        if docs_resp.ok:
            docs = docs_resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"📄 {doc['source_file']}")
                    if col2.button("🗑️", key=f"del_{doc['document_id']}"):
                        requests.delete(f"{API_URL}/documents/{doc['source_file']}")
                        st.rerun()
            else:
                st.info("尚未上傳任何文件")
        else:
            st.warning("無法連線到 API")
    except requests.ConnectionError:
        st.warning("API 未啟動，請先執行 start_api.sh")

# Main area — chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 參考來源"):
                for s in msg["sources"]:
                    page = s.get("page_number")
                    page_str = f" (第{page}頁)" if page else ""
                    st.caption(
                        f"**{s['source_file']}{page_str}** — 相似度: {s['similarity']:.2f}"
                    )
                    st.text(s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"])

if prompt := st.chat_input("請輸入你的問題..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                resp = requests.post(f"{API_URL}/chat", params={"query": prompt})
                if resp.ok:
                    result = resp.json()
                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 參考來源"):
                            for s in sources:
                                page = s.get("page_number")
                                page_str = f" (第{page}頁)" if page else ""
                                st.caption(
                                    f"**{s['source_file']}{page_str}** — 相似度: {s['similarity']:.2f}"
                                )
                                st.text(
                                    s["text"][:200] + "..."
                                    if len(s["text"]) > 200
                                    else s["text"]
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                else:
                    st.error(f"API 錯誤：{resp.status_code}")
            except requests.ConnectionError:
                st.error("無法連線到 API，請確認 API 已啟動")


# COMPLETE TASKS 1-8 EMPLOYEE POLICY ASSISTANT

import streamlit as st
import os, numpy as np, faiss, pandas as pd
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Employee Policy Assistant", layout="wide")

# ---------------- TASK 1 ----------------

policy_documents = {
    "Employee Handbook.pdf":"This employee handbook outlines general company policies, including conduct, dress code, and communication guidelines. All employees are expected to read and adhere to these guidelines. Failure to comply may result in disciplinary action.",
    "Leave Policy.pdf":"Employees are entitled to 12 casual leaves and 15 sick leaves annually. Vacation leave accrues at 1.5 days per month. All leave requests must be submitted through the HR portal at least two weeks in advance, except for emergencies.",
    "Travel Policy.pdf":"Business travel expenses are reimbursed within 30 days of submission. All travel must be pre-approved by a department head. Accommodation and flight bookings should be made through the designated travel portal.",
    "Work From Home Policy.pdf":"Employees may work from home two days per week, subject to managerial approval. A stable internet connection and dedicated workspace are required. Remote work requests should be submitted weekly.",
    "Medical Insurance Policy.pdf":"All full-time employees are covered under the company's comprehensive medical insurance plan. Dependents can be added to the plan at an additional cost. Refer to the insurance handbook for detailed coverage."
}

def create_pdf(filename, content):
    c = canvas.Canvas(filename, pagesize=letter)
    txt = c.beginText(72,720)
    txt.textLines(content)
    c.drawText(txt)
    c.save()

for f,c in policy_documents.items():
    if not os.path.exists(f):
        create_pdf(f,c)

documents=[]
stats=[]

for f in policy_documents:
    reader=PdfReader(f)
    text=""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    documents.append(text)
    stats.append({
        "File Name":f,
        "Pages":len(reader.pages),
        "Characters":len(text),
        "Words":len(text.split())
    })

# ---------------- TASK 2 ----------------

all_text="\n\n".join(documents)

fixed_splitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100
)

fixed_chunks=[x.page_content for x in fixed_splitter.create_documents([all_text])]

recursive_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

recursive_chunks=[x.page_content for x in recursive_splitter.create_documents([all_text])]

# ---------------- TASK 3 ----------------

@st.cache_resource
def load_embed():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model=load_embed()

@st.cache_resource
def get_embeddings():
    return embed_model.encode(recursive_chunks)

embeddings=get_embeddings()

# ---------------- TASK 4 ----------------

dimension=embeddings.shape[1]

index=faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings,dtype=np.float32))

# ---------------- TASK 5 ----------------

def retrieve(query,k=3):
    q=embed_model.encode([query])
    distances,indices=index.search(np.array(q,dtype=np.float32),k)
    return [recursive_chunks[i] for i in indices[0]], distances[0]

# ---------------- TASK 6 ----------------

@st.cache_resource
def load_llm():
    tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-small")
    model=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer,model

tokenizer,llm=load_llm()

# ---------------- TASK 7 ----------------

def rag_answer(question):

    retrieved_chunks,_=retrieve(question)

    context="\n".join(retrieved_chunks)

    prompt=f"""
Answer only from the given context.

Context:
{context}

Question:
{question}
"""

    inputs=tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs=llm.generate(
        **inputs,
        max_new_tokens=60
    )

    answer=tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer,retrieved_chunks

# ---------------- UI ----------------

st.title("Employee Policy Assistant (Tasks 1-8)")

with st.expander("Task 1 - Document Loading"):
    st.dataframe(pd.DataFrame(stats))

with st.expander("Task 2 - Chunking"):
    st.write("Fixed Chunks:",len(fixed_chunks))
    st.write(fixed_chunks[:2])
    st.write("Recursive Chunks:",len(recursive_chunks))
    st.write(recursive_chunks[:2])

with st.expander("Task 3 - Embeddings"):
    st.write("Embedding Shape:",embeddings.shape)
    st.write("Sample Chunk:")
    st.code(recursive_chunks[0])
    st.write("First 5 Embedding Values:")
    st.write(embeddings[0][:5])

with st.expander("Task 4 - FAISS Vector Database"):
    st.write("Number of Chunks:",len(recursive_chunks))
    st.write("Stored Vectors:",index.ntotal)
    st.write("Embedding Dimension:",dimension)

with st.expander("Task 5 - Semantic Retrieval"):
    queries=[
        "How many casual leaves are available?",
        "Can employees work from home?",
        "How does travel reimbursement work?",
        "Who is covered under medical insurance?"
    ]

    for q in queries:
        chunks,_=retrieve(q)
        st.write("Query:",q)
        st.write(chunks[0])
        st.divider()

st.subheader("Task 6 & 7 - Complete RAG Pipeline")

question=st.text_input("Ask a policy question")

if st.button("Ask Question"):
    answer,chunks=rag_answer(question)

    st.success(answer)

    st.subheader("Retrieved Chunks")

    for i,c in enumerate(chunks,1):
        st.write(f"Chunk {i}")
        st.info(c)

with st.expander("Task 8 - Evaluation"):

    evaluation_data=[
        ("How many casual leaves can I take annually?","12"),
        ("What is the policy for sick leave?","15"),
        ("How many days per week can I work from home?","two"),
        ("Who approves travel?","department head"),
        ("What insurance is provided?","medical")
    ]

    results=[]
    correct=0

    for q,expected in evaluation_data:

        response,_=rag_answer(q)

        match=expected.lower() in response.lower()

        if match:
            correct+=1

        results.append({
            "Question":q,
            "Expected":expected,
            "Response":response,
            "Match":match
        })

    st.dataframe(pd.DataFrame(results))

    accuracy=(correct/len(evaluation_data))*100

    st.metric("Overall Accuracy %",round(accuracy,2))

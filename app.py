import os
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from backend.core import ResearchStore, investigate, llm, evidence_graph, available_gemini_models
load_dotenv()
st.set_page_config(page_title='Veritas AI | Research Intelligence',page_icon='🔬',layout='wide')
st.markdown('''<style>.block-container{padding-top:1.5rem}.stApp{background:#0c1220;color:#e8edf6}div[data-testid="stMetric"]{background:#182237;border-radius:12px;padding:14px}</style>''',unsafe_allow_html=True)
st.title('🔬 Veritas AI')
st.caption('Evidence-first multi-stage research intelligence • Hybrid RAG • Claim verification • Contradiction analysis')
@st.cache_resource
def store():return ResearchStore()
kb=store()
with st.sidebar:
    st.header('⚙️ Research settings')
    provider=st.selectbox('LLM provider',['Gemini','Ollama'])
    selected_model=None
    if provider=='Gemini':
        if not os.getenv('GEMINI_API_KEY','').strip():
            st.warning('Add GEMINI_API_KEY to .env and restart Streamlit.')
        if st.button('🔄 Refresh available Gemini models'):
            st.session_state.pop('gemini_models',None)
        if 'gemini_models' not in st.session_state and os.getenv('GEMINI_API_KEY','').strip():
            try:
                with st.spinner('Checking Gemini model access...'):
                    st.session_state['gemini_models']=available_gemini_models()
            except Exception as exc:
                st.warning(f'Could not list models: {exc}')
        models=st.session_state.get('gemini_models',[])
        preferred=os.getenv('GEMINI_MODEL','gemini-2.5-flash').removeprefix('models/')
        if models:
            idx=models.index(preferred) if preferred in models else next((i for i,m in enumerate(models) if 'flash' in m.lower()),0)
            selected_model=st.selectbox('Gemini model (available to your key)',models,index=idx)
        else:
            selected_model=st.text_input('Gemini model ID',value=preferred)
    else:
        selected_model=st.text_input('Ollama model',value=os.getenv('OLLAMA_MODEL','gemma3:4b'))
    st.caption('Gemini runs in the cloud. Ollama runs locally.')
    k=st.slider('Retrieved evidence passages',3,15,8)
    st.divider()
    st.subheader('📚 Knowledge base')
    st.metric('Indexed chunks',kb.collection.count())
    st.metric('Documents',len(kb.list_sources()))
    st.caption('First embedding run downloads a local model (~90 MB).')
tabs=st.tabs(['📥 Library','🧪 Investigation','🛡️ Verification','⚖️ Contradictions','🕸️ Knowledge Graph','📊 Evaluation'])
with tabs[0]:
    st.subheader('Index research documents')
    files=st.file_uploader('Upload research papers (PDF, DOCX, TXT)',type=['pdf','docx','txt'],accept_multiple_files=True)
    if st.button('Index uploaded documents',type='primary',disabled=not files):
        for f in files:
            try:
                with st.spinner(f'Indexing {f.name}...'):
                    n=kb.add_file(f.name,f.getvalue())
                st.success(f'{f.name}: {n} text chunks indexed')
            except Exception as exc:st.error(f'{f.name}: {exc}')
    st.dataframe(pd.DataFrame([{'Document':n,'Chunks':c} for n,c in kb.list_sources().items()]),use_container_width=True,hide_index=True)
with tabs[1]:
    st.subheader('Autonomous-style research investigation')
    question=st.text_area('Research question','Compare the methods and evaluation approaches described in the uploaded papers.',height=95,key='investigation_question')
    st.caption('Gemini requests usually complete faster than local CPU inference; retrieval still runs on your computer.')
    if st.button('Run investigation',type='primary'):
        try:
            with st.spinner('Retrieving, synthesizing and citing evidence...'):
                result=investigate(kb,question,provider,'investigate',k,model=selected_model)
            st.session_state['last_result']=result
            st.markdown(result['answer']);st.caption(f"Completed in {result['latency']} seconds")
        except Exception as exc:st.error(str(exc))
    if 'last_result' in st.session_state:
        r=st.session_state['last_result']
        st.download_button('⬇️ Download research report (.md)',r['answer'],file_name='veritas_research_report.md',mime='text/markdown')
        with st.expander('Inspect retrieved evidence'):
            for i,p in enumerate(r['sources'],1):st.markdown(f"**[{i}] {p['source']} · Page {p['page']} · Score {p['score']}**\n\n{p['text']}")
with tabs[2]:
    st.subheader('Claim verification')
    claim=st.text_area('Claim to check','The proposed models were evaluated on independent test datasets.',key='claim')
    if st.button('Verify claim'):
        try:
            with st.spinner('Checking evidence...'):r=investigate(kb,claim,provider,'verify',k,model=selected_model)
            st.markdown(r['answer'])
            with st.expander('Evidence'): 
                for i,p in enumerate(r['sources'],1):st.write(f"[{i}] {p['source']} p.{p['page']}: {p['text']}")
        except Exception as exc:st.error(str(exc))
with tabs[3]:
    st.subheader('Cross-paper contradiction detection')
    topic=st.text_input('Topic or comparison','Model performance and limitations')
    if st.button('Analyze contradictions'):
        try:
            with st.spinner('Comparing evidence...'):r=investigate(kb,topic,provider,'contradictions',k,model=selected_model)
            st.markdown(r['answer'])
        except Exception as exc:st.error(str(exc))
with tabs[4]:
    st.subheader('Evidence knowledge graph')
    query=st.text_input('Graph evidence related to','machine learning models')
    if st.button('Build graph'):
        import networkx as nx
        evidence=kb.search(query,k=15); g=evidence_graph(evidence)
        if not g.nodes:st.info('No evidence available.')
        else:
            pos=nx.spring_layout(g,seed=42)
            ex=[];ey=[]
            for a,b in g.edges():ex.extend([pos[a][0],pos[b][0],None]);ey.extend([pos[a][1],pos[b][1],None])
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=ex,y=ey,mode='lines',line=dict(width=1,color='#63728b'),hoverinfo='none'))
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in g.nodes],y=[pos[n][1] for n in g.nodes],mode='markers+text',text=list(g.nodes),textposition='top center',marker=dict(size=[20 if g.nodes[n]['kind']=='document' else 12 for n in g.nodes],color=['#46d5ca' if g.nodes[n]['kind']=='document' else '#f9bf69' for n in g.nodes]),textfont=dict(color='white')))
            fig.update_layout(height=550,showlegend=False,paper_bgcolor='#0c1220',plot_bgcolor='#0c1220',xaxis=dict(visible=False),yaxis=dict(visible=False))
            st.plotly_chart(fig,use_container_width=True)
            st.caption('Graph links document names to selected model/metric terms appearing in retrieved text; it does not assert causal relationships.')
with tabs[5]:
    st.subheader('Retrieval evaluation workspace')
    st.info('Compare retrieval results against manually labeled relevant passages. LLM-generated judgments are not ground-truth accuracy measurements.')
    q=st.text_input('Evaluation query','What models were compared?')
    if st.button('Inspect retrieval ranking'):
        rows=kb.search(q,k=10)
        st.dataframe(pd.DataFrame([{'Rank':i,'Source':p['source'],'Page':p['page'],'RRF score':p['score'],'Excerpt':p['text'][:180]} for i,p in enumerate(rows,1)]),hide_index=True,use_container_width=True)
    st.markdown('**Research methodology:** Compare standard dense retrieval against hybrid RRF on a labeled query set; report Recall@K, citation precision, answer faithfulness, latency, and human-reviewed contradiction accuracy.')

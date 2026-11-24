from backend.core import chunk_text, read_document, evidence_prompt

def test_chunking_overlap():
    chunks=chunk_text('word '*1000,size=100,overlap=20)
    assert len(chunks)>1 and all(len(c)<=100 for c in chunks)

def test_txt():
    assert read_document('sample.txt',b'hello')[0]==(1,'hello')

def test_prompt_cites():
    prompt=evidence_prompt('What?',[{'source':'test.pdf','page':2,'text':'Evidence.'}])
    assert '[1]' in prompt and 'test.pdf page 2' in prompt

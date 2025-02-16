import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import spacy
import nltk
from nltk.corpus import cmudict

nltk.download('cmudict')

nlp = spacy.load("en_core_web_sm")

@st.cache_resource
def load_model():
    model_name = "gpt2-poetry"  
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()  
    return tokenizer, model

tokenizer, model = load_model()

cmu_dict = cmudict.dict()

def get_last_syllable(word):
    word = word.lower()
    if word in cmu_dict:
        pronunciations = cmu_dict[word]
        return pronunciations[0][-1]
    else:
        return None

def check_rhyme(line1, line2):
    try:
        word1 = line1.strip().split()[-1]
        word2 = line2.strip().split()[-1]
    except IndexError:
        return False
    syllable1 = get_last_syllable(word1)
    syllable2 = get_last_syllable(word2)
    return syllable1 is not None and syllable2 is not None and syllable1 == syllable2

st.title("AI Poetry Generator")

prompt = st.text_area("Enter a prompt for your poem", "Once upon a midnight dreary,")

num_return_sequences = st.slider("Number of poem variations", min_value=1, max_value=5, value=1)
max_length = st.slider("Max length of generated poem", min_value=50, max_value=300, value=100)

if st.button("Generate Poetry"):
    st.write("Generation started...")
    with st.spinner("Generating your poem. Please wait..."):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        poems = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    st.write("Generation finished!")

    for i, poem in enumerate(poems):
        st.subheader(f"Poem Variation {i+1}")
        st.write(poem)
        
        st.write("**Rhyme Analysis:**")
        lines = poem.split("\n")
        if len(lines) < 2:
            st.write("Not enough lines for rhyme analysis.")
        else:
            for j in range(len(lines) - 1):
                if lines[j].strip() and lines[j+1].strip():
                    rhyme = check_rhyme(lines[j], lines[j+1])
                    last_word1 = lines[j].split()[-1]
                    last_word2 = lines[j+1].split()[-1]
                    st.write(f"'{last_word1}' and '{last_word2}' rhyme? {'Yes' if rhyme else 'No'}")
        
        st.write("**Named Entities Detected (spaCy):**")
        doc = nlp(poem)
        if doc.ents:
            for ent in doc.ents:
                st.write(f"{ent.text} ({ent.label_})")
        else:
            st.write("No named entities found.")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import spacy
import nltk
from nltk.corpus import cmudict

nltk.download('cmudict')

app = FastAPI(title="AI Poetry Generator API")

nlp = spacy.load("en_core_web_sm")

model_name = "gpt2-poetry"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

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

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    num_return_sequences: int = 1

class GenerationResponse(BaseModel):
    generated_poems: list
    rhyme_analysis: list

@app.post("/generate", response_model=GenerationResponse)
def generate_poetry(request: GenerationRequest):
    try:
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
        outputs = model.generate(
            input_ids,
            max_length=request.max_length,
            num_return_sequences=request.num_return_sequences,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
        poems = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        analysis = []
        for poem in poems:
            lines = poem.split("\n")
            poem_analysis = []
            for j in range(len(lines) - 1):
                if lines[j].strip() and lines[j+1].strip():
                    rhyme = check_rhyme(lines[j], lines[j+1])
                    poem_analysis.append({
                        "line1_last_word": lines[j].split()[-1],
                        "line2_last_word": lines[j+1].split()[-1],
                        "rhyme": rhyme
                    })
            analysis.append(poem_analysis)
            
        return GenerationResponse(generated_poems=poems, rhyme_analysis=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

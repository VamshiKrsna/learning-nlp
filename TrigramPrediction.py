import nltk 
from nltk.corpus import reuters
from nltk import bigrams, trigrams 
from collections import Counter, defaultdict
import torch 
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

# Download corpora and tagger
nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


model = defaultdict(lambda: defaultdict(lambda: 0))

# frequency of co-occurences
for sentence in reuters.sents():
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1



for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
        # print(total_count)
        # print(model[w1_w2][w3])

# print(model)

sorted_probabilities = sorted(dict(model["the","price"]).items(), key=lambda x: x[1], reverse=True)

print("Most Probable Words following 'the price', in order:")
for word, prob in sorted_probabilities:
    print(f"{word}: {prob:.4f}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# reproducing text completion by gpt2

input_text = "Once Upon A Time, There Lived a"

print(f"Input Text : {input_text}")

#encode input : 
encoded_ip = tokenizer.encode(input_text)
# gen tensor for encoded input
ip_tensor = torch.tensor([encoded_ip])
# load gpt2
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.eval() # eval mode to deactivate DropOut modules

ip_tensor = ip_tensor.to('cpu')
model.to('cpu')

with torch.no_grad():
    outputs = model(ip_tensor)
    preds = outputs[0]

predicted_idx = torch.argmax(preds[0,-1,:]).item()
predicted_token = tokenizer.decode(predicted_idx)

pred_text = input_text + predicted_token

print(f"Predicted next token : {predicted_token}")
print(f"Complete prediction : {pred_text}")

# Generating next n no. of tokens (to form a sentence)

num_toks_to_gen = 10
gen_text = input_text 

for i in range(num_toks_to_gen):
    input_ids = tokenizer.encode(gen_text)
    tokens_tensor = torch.tensor([input_ids]).to('cpu')

    with torch.no_grad():
        ops = model(tokens_tensor)
        preds = ops[0]
    
    predicted_idx = torch.argmax(preds[0,-1,:]).item()
    predicted_token = tokenizer.decode(predicted_idx)
    
    # Generated text 
    gen_text += predicted_token

print(f"Generated Text with {num_toks_to_gen} new tokens : {gen_text}")
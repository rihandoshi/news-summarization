import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset

dataset = load_dataset("xsum")

print(torch.version.cuda)
print(torch.backends.cuda.is_built())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def attention(q, k, v, mask = None):
    d = q.shape[-1]
    attention = q @ k.transpose(-2,-1) / (d ** 0.5)

    if mask is not None:
        attention = attention.masked_fill(mask == 0, -1e9)

    attention = torch.softmax(attention, dim = -1)
    # print(attention[:, :, -1])
    output = attention @ v
    return output

# q = torch.randn(2, 5, 16)
# k = torch.randn(2, 5, 16)
# v = torch.randn(2, 5, 16)

# out = attention(q, k, v)

# mask = torch.ones(2, 5, 5)
# mask[:, :, -1] = 0

# out = attention(q, k, v, mask)



    
# x = torch.randn(2, 5, 16)
# out = multi_head_attention(x)

# print(out.shape)

# mask = torch.ones(2, 5, 5)
# mask[:, :, -1] = 0  # block last token

# out = multi_head_attention(x, mask)
# print(out.shape)

# out1 = multi_head_attention(x)
# out2 = multi_head_attention(x)

# print(torch.allclose(out1, out2))

# def encoder(x, mask=None):
#     B, T, D = x.shape
#     output1 = multi_head_attention(x, mask)

#     output2 = x + output1
#     ln1 = torch.nn.LayerNorm(D)

#     output2 = ln1(output2)

#     W1 = torch.randn(D, 4 * D)
#     W2 = torch.randn(4 * D, D)

#     ffout = torch.relu(output2 @ W1) @ W2
#     output3 = output2 + ffout
#     ln2 = torch.nn.LayerNorm(D)
#     output3 = ln2(output3)

#     return output3


class Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.W1 = nn.Linear(d_model, 4 * d_model)
        self.W2 = nn.Linear(4 * d_model, d_model)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.number_of_heads = 4
        self.head_size = d_model // self.number_of_heads

    
    def forward(self, x, mask=None):
        attention_output = self.multi_head_attention(x, mask)

        x = x + attention_output
        x = self.ln1(x)

        ff_output = torch.relu(self.W1(x))
        ff_output = self.W2(ff_output)

        x = x + ff_output
        x = self.ln2(x)
        return x
    

    def multi_head_attention(self, x, mask=None):
        B, T , D = x.shape
        number_of_heads = self.number_of_heads
        head_size = self.head_size

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q = q.view(B, T, number_of_heads, head_size).transpose(1,2)
        k = k.view(B, T, number_of_heads, head_size).transpose(1,2)
        v = v.view(B, T, number_of_heads, head_size).transpose(1,2)

        attention = q @ k.transpose(-2, -1) / (head_size ** 0.5)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(attention, dim=-1)

        out = attention @ v
        out = out.transpose(1, 2)
        out = out.contiguous().view(B, T, D)

        out = self.Wo(out)
        return out
        

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, max_len=1000, vocab_size=1000):
        super().__init__()
        self.layers = nn.ModuleList([Encoder(d_model) for _ in range(num_layers)])
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_len, d_model)


    def forward(self, x, mask=None):
        x = self.embedding(x)
        B, T, D = x.shape

        positions = torch.arange(T)
        positions = positions.to(x.device)
        positions = positions.unsqueeze(0).expand(B, T)

        position_embed = self.positional_encoding(positions)
        
        x = x + position_embed
        for layer in self.layers:
            x = layer(x, mask)

        return x

    
class Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.W1 = nn.Linear(d_model, 4 * d_model)
        self.W2 = nn.Linear(4 * d_model, d_model)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.cWq = nn.Linear(d_model, d_model)
        self.cWk = nn.Linear(d_model, d_model)
        self.cWv = nn.Linear(d_model, d_model)

        self.number_of_heads = 4
        self.head_size = d_model // self.number_of_heads

    
    def forward(self, x, encoder_output, mask=None, encoder_padding_mask=None):
        B, T , D = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q = q.view(B, T, self.number_of_heads, self.head_size).transpose(1,2)
        k = k.view(B, T, self.number_of_heads, self.head_size).transpose(1,2)
        v = v.view(B, T, self.number_of_heads, self.head_size).transpose(1,2)

        masked_attention_output = self.multi_head_attention(x, q, k, v, mask)

        x = x + masked_attention_output
        x = self.ln1(x)

        x = x + self.cross_attention(x, encoder_output, encoder_padding_mask)
        x = self.ln2(x)

        # Here is the feed forward network now.
        ff_output = torch.relu(self.W1(x))
        ff_output = self.W2(ff_output)

        x = x + ff_output
        x = self.ln3(x)
        return x
    

    def multi_head_attention(self, x, q, k, v, mask=None):
        B, T , D = x.shape
        head_size = self.head_size
        # q = q.view(B, T, self.number_of_heads, self.head_size).transpose(1,2)
        # k = k.view(B, T, self.number_of_heads, self.head_size).transpose(1,2)
        # v = v.view(B, T, self.number_of_heads, self.head_size).transpose(1,2)
        attention = q @ k.transpose(-2, -1) / (head_size ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(attention, dim=-1)

        out = attention @ v
        out = out.transpose(1, 2)
        out = out.contiguous().view(B, T, D)

        out = self.Wo(out)
        return out
    
    def cross_attention(self, x, encoder_output, mask=None):
        q = self.cWq(x)
        k = self.cWk(encoder_output)
        v = self.cWv(encoder_output)
        B, T_dec, D = x.shape
        T_enc = encoder_output.shape[1]

        q = q.view(B, T_dec, self.number_of_heads, self.head_size).transpose(1,2)
        k = k.view(B, T_enc, self.number_of_heads, self.head_size).transpose(1,2)
        v = v.view(B, T_enc, self.number_of_heads, self.head_size).transpose(1,2)
        output = self.multi_head_attention(x, q, k, v, mask)
        return output
    

# x = torch.randn(2, 5, 16)

# model = Encoder(16)

# out = model(x)

# print(out.shape)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_layers, max_len=1000, vocab_size=1000):
        super().__init__()
        self.layers = nn.ModuleList([Decoder(d_model) for _ in range(num_layers)])
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_len, d_model)

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, mask=None, encoder_padding_mask=None):
        x = self.embedding(x)
        B, T, D = x.shape

        mask = torch.tril(torch.ones(T, T)).to(x.device)
        positions = torch.arange(T)
        positions = positions.to(x.device)
        positions = positions.unsqueeze(0).expand(B, T)

        position_embed = self.positional_encoding(positions)
        
        x = x + position_embed
        for layer in self.layers:
            x = layer(x, encoder_output, mask, encoder_padding_mask)

        logits = self.output_layer(x)

        return logits
    

    

char_to_int = dict()
int_to_char = dict()

training_pairs = [
    (x["document"][:300], x["summary"][:80])
    for x in dataset["train"].select(range(10000))
]

articles = [pair[0] for pair in training_pairs]
summaries = [pair[1] for pair in training_pairs]
training_string = " ".join(articles + summaries)
i = 0
# for c in training_string:
#     if c not in int_to_char.values():
#         int_to_char[i] = c
#         i += 1

# i = 0
# for c in training_string:
#     if c not in char_to_int.keys():
#         char_to_int[c] = i
#         i += 1

split_string = [list(training_string)]
corpus = split_string
adjacent_pair_counts = dict()

def get_pair_counts(sentences):
    pair_counts = {}
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            pair = (sentence[i], sentence[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def merge_pair(corpus, pair):
    new_token = "".join(pair)

    for sentence in corpus:
        i = 0
        while i < len(sentence) - 1:
            if (sentence[i], sentence[i+1]) == pair:
                sentence[i] = new_token
                del sentence[i+1]
            else:
                i += 1

    return corpus

num_merges = 6000
merges = []

for _ in range(num_merges):
    pair_counts = get_pair_counts(corpus)
    if not pair_counts:
        break

    best_pair = max(pair_counts, key=pair_counts.get)
    merges.append(best_pair)

    corpus = merge_pair(corpus, best_pair)

vocab = set()
for sentence in corpus:
    for token in sentence:
        vocab.add(token)

vocab = set(vocab)
vocab.add("<PAD>")
vocab.add("<START>")
vocab.add("<END>")
vocab.add("<MASK>")
vocab_size = len(vocab)

token_to_int = {tok: i for i, tok in enumerate(vocab)}
int_to_token = {i: tok for tok, i in token_to_int.items()}

def encode(text, merges):
    tokens = list(text)

    for pair in merges:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == pair:
                tokens[i] = tokens[i] + tokens[i+1]
                del tokens[i+1]
            else:
                i += 1

    return tokens


encoder = TransformerEncoder(d_model=128, num_layers=3, vocab_size=vocab_size).to(device)
decoder = TransformerDecoder(d_model=128, num_layers=3, vocab_size=vocab_size).to(device)
optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()))
loss_function = nn.CrossEntropyLoss(ignore_index=token_to_int["<PAD>"])
batch_size = 32
tokenized_pairs = [
    (encode(a, merges), ["<START>"] + encode(s, merges) + ["<END>"])
    for a, s in training_pairs
]

val_pairs = [
    (x["document"][:300], x["summary"][:80])
    for x in dataset["validation"].select(range(500))
]

val_tokenized = [
    (encode(a, merges), ["<START>"] + encode(s, merges) + ["<END>"])
    for a, s in val_pairs
]

def validate():
    encoder.eval()
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        idx = torch.randint(0, len(val_tokenized), (batch_size,))

        batch_articles = [val_tokenized[i][0] for i in idx]
        batch_summaries = [val_tokenized[i][1] for i in idx]

        max_len_article = max(len(a) for a in batch_articles)
        max_len_summary = max(len(s) for s in batch_summaries)
        
        

        # We should add padding <PAD> token at the end and also add <START> and <END> tokens to the summary.

        encoded_articles = [
            x + ["<PAD>"] * (max_len_article - len(x))
            for x in batch_articles
        ]

        encoded_summaries = [
            x + ["<PAD>"] * (max_len_summary - len(x))
            for x in batch_summaries
        ]
        article = torch.tensor([[token_to_int[t] if t in token_to_int else token_to_int["<PAD>"] for t in seq] for seq in encoded_articles]).to(device)
        summary = torch.tensor([[token_to_int[t] if t in token_to_int else token_to_int["<PAD>"] for t in seq] for seq in encoded_summaries]).to(device)
        pad_mask = (article != token_to_int["<PAD>"]).to(device)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        encoder_output = encoder(article, pad_mask)
        decoder_input = summary[:, :-1]

            # randomly mask some tokens in the decoder input for teacher forcing
        mask = torch.rand(decoder_input.shape) < 0.1
        mask = mask.to(device)
        decoder_input = decoder_input.masked_fill(mask, token_to_int["<MASK>"])

        target = summary[:, 1:]
        logits = decoder(decoder_input, encoder_output, None, pad_mask)
        loss = loss_function(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1)
        )
        loss = loss.item()
        total_loss += loss
    return total_loss


encoder.train()
decoder.train()

for step in range(8000):
    idx = torch.randint(0, len(training_pairs), (batch_size,))

    batch_articles = [tokenized_pairs[i][0] for i in idx]
    batch_summaries = [tokenized_pairs[i][1] for i in idx]

    max_len_article = max(len(a) for a in batch_articles)
    max_len_summary = max(len(s) for s in batch_summaries)
    
    

    # We should add padding <PAD> token at the end and also add <START> and <END> tokens to the summary.

    encoded_articles = [
        x + ["<PAD>"] * (max_len_article - len(x))
        for x in batch_articles
    ]

    encoded_summaries = [
        x + ["<PAD>"] * (max_len_summary - len(x))
        for x in batch_summaries
    ]
    article = torch.tensor([[token_to_int[t] if t in token_to_int else token_to_int["<PAD>"] for t in seq] for seq in encoded_articles]).to(device)
    summary = torch.tensor([[token_to_int[t] if t in token_to_int else token_to_int["<PAD>"] for t in seq] for seq in encoded_summaries]).to(device)
    pad_mask = (article != token_to_int["<PAD>"]).to(device)
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

    encoder_output = encoder(article, pad_mask)
    decoder_input = summary[:, :-1]
    # randomly mask some tokens in the decoder input for teacher forcing
    mask = torch.rand(decoder_input.shape) < 0.1
    mask = mask.to(device)
    decoder_input = decoder_input.masked_fill(mask, token_to_int["<MASK>"])
    target = summary[:, 1:]
    logits = decoder(decoder_input, encoder_output, None, pad_mask)
    loss = loss_function(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        val_loss = validate()
        print("VAL:", val_loss)
        encoder.train()
        decoder.train()
# This predicts the next token at once. But we should predict one token at a time.
# input_ids = torch.tensor([[20, 21, 22, 23]])
# logits = model(input_ids)

# pred = torch.argmax(logits, dim=-1)
# print(pred)

### One token at a time loop




temperature = 0.8
test_article = training_pairs[0][0]

tokens = encode(test_article, merges)
article = torch.tensor([[token_to_int.get(t, token_to_int["<PAD>"]) for t in tokens]]).to(device)




def generate(article):
    encoder.eval()
    decoder.eval()
    summary_ids = torch.tensor([[token_to_int["<START>"]]]).to(device)
    

    with torch.no_grad():
        pad_mask = (article != token_to_int["<PAD>"]).to(device)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        encoder_output = encoder(article, pad_mask).to(device)
        generated_string = ""
        for _ in range(50):
            logits = decoder(summary_ids, encoder_output, None, pad_mask)

            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits / temperature, dim=-1)

            top_k = 5
            topk_probs, topk_indices = torch.topk(probs, top_k)

            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            sampled_index = torch.multinomial(topk_probs, 1)
            next_token = topk_indices.gather(-1, sampled_index)
            
            generated_string += int_to_token[next_token.item()]
            summary_ids = torch.cat([summary_ids, next_token], dim=1)
            if next_token.item() == token_to_int["<END>"]:
                break
    return generated_string


generated = generate(article)
print("ARTICLE:", test_article)
print("GENERATED:", generated)
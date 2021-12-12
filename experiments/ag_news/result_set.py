import torch
import captum
# from train_without_embedding_bag import TextClassificationModel
from tc_lstm import LSTMClassifier
from torchtext.datasets import AG_NEWS
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from captum.attr import LimeBase, KernelShap,InputXGradient
from captum._utils.models.linear_model import SkLearnLasso
import torch.nn.functional as F
from IPython.core.display import HTML, display
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer, GradientShap
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization, IntegratedGradients
# import shap
from scipy.stats.stats import pearsonr
import itertools


### TODO
# 1. Compute all the explanations on all the test samples and average the rank correlation maps for them.
# 2. Compute top-K rank correlations.
# 3. Have a custom map to pay with K of the previous metric.

NUM_EXPLANATIONS = 3
EXPLANATIONS = dict()
EXPLANATION_NAMES = dict()
pointer = 0

def num_to_text(text_nums, vocab) :
    return [vocab.vocab.itos_[i] for i in text_nums]


PATH = "./text_classification_lstm2.model"
tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(x)
label_pipeline = lambda x: int(x) - 1


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def append_pads(text, max_len):
    text.extend(['<pad>']*(max_len-len(text)))

def collate_batch(batch):
    batch = [(i[0], tokenizer(i[1]) ) for i in batch]
    max_len = max([len(i[1]) for i in batch])
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        append_pads(_text, max_len)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.stack(text_list)
    return label_list.to(device), text_list.to(device)


train_iter = AG_NEWS(split='train')
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
#model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
model = torch.load(PATH, map_location=torch.device('cpu'))
import time


# In[5]:


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


# In[6]:


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

# Hyperparameters
EPOCHS = 1  # epoch
LR = 5  # learning rate
BATCH_SIZE = 1  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = AG_NEWS()

train_dataset = to_map_style_dataset(train_iter)

test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ =     random_split(train_dataset, [num_train, len(train_dataset) - num_train])


test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)
print("The accuracy of the trained models : {}".format(evaluate(test_dataloader)))







# Explanation Method : 1 : Lime Code
# remove the batch dimension for the embedding-bag model
def forward_func(text):
    return model(text)

# encode text indices into latent representations & calculate cosine similarity
def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
    original_emb = torch.mean(model.word_embeddings(original_inp), dim = -2)
    perturbed_emb = torch.mean(model.word_embeddings(perturbed_inp), dim = -2)
    #print("-->>", original_emb.shape, perturbed_emb.shape, model.embedding(original_inp).shape)
    distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
    return torch.exp(-1 * (distance ** 2) / 2)

# binary vector where each word is selected independently and uniformly at random
def bernoulli_perturb(text, **kwargs):
    probs = torch.ones_like(text) * 0.5
    return torch.bernoulli(probs).long()

# remove abscent token based on the intepretable representation sample
def interp_to_input(interp_sample, original_input, **kwargs):
    return original_input[interp_sample.bool()].view(original_input.size(0), -1)



lasso_lime_base = LimeBase(
    forward_func,
    interpretable_model=SkLearnLasso(alpha=0.08),
    similarity_func=exp_embedding_cosine_distance,
    perturb_func=bernoulli_perturb,
    perturb_interpretable_space=True,
    from_interp_rep_transform=interp_to_input,
    to_interp_rep_transform=None
)

test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
             '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
             'And it is too early to tell if he will match Aleksandr Dityatin, '
             'the Soviet gymnast who won eight total medals in 1980.')

test_labels, test_text = collate_batch([(test_label, test_line)])
print(test_text, model(test_text))

probs = F.softmax(model(test_text), dim=1).squeeze(0)
print('Prediction probability:', round(probs[test_labels[0]].item(), 4), probs)

attrs_laso = lasso_lime_base.attribute(
    test_text,  # add batch dimension for Captum
    target=test_labels,
    n_samples=32000,
    show_progress=True
).squeeze(0)
#attrs = F.normalize(attrs, p=2.0, dim=0, eps=1e-12, out=None)

#print(attrs)
def show_text_attr(attrs):
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    alpha = lambda x: abs(x) ** 0.5
    token_marks = [
        f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
        for token, attr in zip(tokenizer(test_line), attrs.tolist())
    ]

    display(HTML('<p>' + ' '.join(token_marks) + '</p>'))



# show_text_attr(attrs_laso)

EXPLANATIONS[pointer] = attrs_laso.detach().numpy()
EXPLANATION_NAMES['LI'] = pointer
pointer += 1


# Explanation Method 2 :  KernalSHAP


test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
                 '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
                 'And it is too early to tell if he will match Aleksandr Dityatin, '
                 'the Soviet gymnast who won eight total medals in 1980.')

test_labels, test_text = collate_batch([(test_label, test_line)])


interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')


def construct_whole_bert_embeddings(input_ids, ref_input_ids,                                         token_type_ids=None, ref_token_type_ids=None,                                         position_ids=None, ref_position_ids=None):
        input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
        ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
        return input_embeddings, ref_input_embeddings

layer_grad_shap = KernelShap(model) 
baselines = torch.zeros(test_text.shape[0]).to(torch.int64)
input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(test_text, baselines)
attribution_shap = torch.sum(layer_grad_shap.attribute(input_embeddings, ref_input_embeddings,
                                            target=test_labels)*input_embeddings, dim = -1)

attribution_shap = attribution_shap *100
EXPLANATIONS[pointer] = attribution_shap[0].detach().numpy()
EXPLANATION_NAMES['SH'] = pointer
pointer += 1

def show_text_attr(attrs):
        rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
        alpha = lambda x: abs(x) ** 0.5
        token_marks = [
            f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
            for token, attr in zip(tokenizer(test_line), attrs.tolist())
        ]    
        display(HTML('<p>' + ' '.join(token_marks) + '</p>'))
    
# show_text_attr(attribution_shap.squeeze(0))

# Explanation Method 3 : Integrated Gradients


def construct_whole_bert_embeddings(input_ids, ref_input_ids, token_type_ids=None, ref_token_type_ids=None, position_ids=None, ref_position_ids=None):
        input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
        ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
        return input_embeddings, ref_input_embeddings

from captum.attr import IntegratedGradients, TokenReferenceBase, visualization

token_reference = TokenReferenceBase(reference_token_idx=0)
vis_data_records_ig = []

def interpret_sentence(model, test_text, test_labels,  pred, pred_ind, min_len=7, interpretable_embedding = None, label=0):
        model.zero_grad()
        # input_indices dim: [sequence_length]
        seq_length = test_text.shape[0]
        # generate reference indices for each sample
        
        reference_indices = token_reference.generate_reference(seq_length, device=device)
        input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(test_text, reference_indices, interpretable_embedding)
        # compute attributions and approximation delta using layer integrated gradients
        ig = IntegratedGradients(model)
        attributions_ig = ig.attribute(input_embeddings, ref_input_embeddings, #additional_forward_args=(test_offsets,),\
                                               n_steps=500, target=test_labels)
        return attributions_ig #*input_embeddings

model = torch.load(PATH, map_location=torch.device('cpu'))
test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
                 '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
                 'And it is too early to tell if he will match Aleksandr Dityatin, '
                 'the Soviet gymnast who won eight total medals in 1980.')
test_labels, test_text = collate_batch([(test_label, test_line)])
pred = F.softmax(model(test_text), dim=1)
pred_ind = torch.round(pred)
interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')
attrs_ig = interpret_sentence(model, test_text, test_labels,  pred,pred_ind,  label=2, interpretable_embedding=interpretable_embedding)
attrs_ig2 = torch.sum(attrs_ig, dim = -1)

EXPLANATIONS[pointer] = attrs_ig2[0].detach().numpy()
EXPLANATION_NAMES['IG'] = pointer
pointer += 1
# show_text_attr(attrs_ig2[0])

# Explanation Method 4 : Smooth Grad


from captum.attr import NoiseTunnel
from captum.attr import Saliency


#
# def construct_whole_bert_embeddings(input_ids, ref_input_ids,                                         token_type_ids=None, ref_token_type_ids=None,                                         position_ids=None, ref_position_ids=None):
#         input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
#         ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
#         return input_embeddings, ref_input_embeddings

token_reference = TokenReferenceBase(reference_token_idx=0)
vis_data_records_ig = []


def interpret_sentence(model, test_text, test_labels, pred, pred_ind, min_len=7, interpretable_embedding=None, label=0, num_samples=500):
    # model.eval()
    model.zero_grad()
    seq_length = test_text.shape[0]
    # generate reference indices for each sample

    noise_tunnel = NoiseTunnel(Saliency(model))
    input_embeddings = interpretable_embedding.indices_to_embeddings(test_text)
    print(input_embeddings.shape)
    attribution = noise_tunnel.attribute(input_embeddings,
                                         nt_type='smoothgrad',
                                         target=2,
                                         nt_samples=num_samples)
    return attribution  # *input_embeddings

model = torch.load(PATH, map_location=torch.device('cpu'))
test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
                 '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
                 'And it is too early to tell if he will match Aleksandr Dityatin, '
                 'the Soviet gymnast who won eight total medals in 1980.')
test_labels, test_text = collate_batch([(test_label, test_line)])
pred = F.softmax(model(test_text), dim=1)
pred_ind = torch.round(pred)
interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')
attrs_smg = interpret_sentence(model, test_text, test_labels,  pred,pred_ind,  label=2, interpretable_embedding=interpretable_embedding, num_samples = 500)
attrs_smg2 = torch.sum(attrs_smg, dim = -1)

EXPLANATIONS[pointer] = attrs_smg2[0].detach().numpy()
EXPLANATION_NAMES['SG'] = pointer
pointer += 1
# show_text_attr(attrs_smg2[0])

# Explanation Method 5 : Grad
token_reference = TokenReferenceBase(reference_token_idx=0)
vis_data_records_ig = []


def interpret_sentence(model, test_text, test_labels, pred, pred_ind, min_len=7, interpretable_embedding=None, label=0, num_samples=500):
    # model.eval()
    model.zero_grad()
    seq_length = test_text.shape[0]
    # generate reference indices for each sample

    saliency_map = Saliency(model)
    input_embeddings = interpretable_embedding.indices_to_embeddings(test_text)
    print(input_embeddings.shape)
    attribution = saliency_map.attribute(input_embeddings, target=2)
    # saliency.attribute(input, target=3)
    return attribution  # *input_embeddings

model = torch.load(PATH, map_location=torch.device('cpu'))
test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
                 '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
                 'And it is too early to tell if he will match Aleksandr Dityatin, '
                 'the Soviet gymnast who won eight total medals in 1980.')
test_labels, test_text = collate_batch([(test_label, test_line)])
pred = F.softmax(model(test_text), dim=1)
pred_ind = torch.round(pred)
interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')
attrs_gr = interpret_sentence(model, test_text, test_labels,  pred,pred_ind,  label=2, interpretable_embedding=interpretable_embedding, num_samples = 500)
attrs_gr2 = torch.sum(attrs_gr, dim = -1)

EXPLANATIONS[pointer] = attrs_gr2[0].detach().numpy()
EXPLANATION_NAMES['Grad'] = pointer
pointer += 1
# show_text_attr(attrs_gr2[0])

# Explanation Method 6 : Gradients * Inputs

token_reference = TokenReferenceBase(reference_token_idx=0)
vis_data_records_ig = []


def interpret_sentence(model, test_text, test_labels, pred, pred_ind, min_len=7, interpretable_embedding=None, label=0, num_samples=500):
    # model.eval()
    model.zero_grad()
    seq_length = test_text.shape[0]
    # generate reference indices for each sample

    saliency_map = InputXGradient(model)
    input_embeddings = interpretable_embedding.indices_to_embeddings(test_text)
    print(input_embeddings.shape)
    attribution = saliency_map.attribute(input_embeddings, target=2)
    # saliency.attribute(input, target=3)
    return attribution  # *input_embeddings

model = torch.load(PATH, map_location=torch.device('cpu'))
test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
                 '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
                 'And it is too early to tell if he will match Aleksandr Dityatin, '
                 'the Soviet gymnast who won eight total medals in 1980.')
test_labels, test_text = collate_batch([(test_label, test_line)])
pred = F.softmax(model(test_text), dim=1)
pred_ind = torch.round(pred)
interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')
attrs_gri = interpret_sentence(model, test_text, test_labels,  pred,pred_ind,  label=2, interpretable_embedding=interpretable_embedding, num_samples = 500)
attrs_gri2 = torch.sum(attrs_gri, dim = -1)

EXPLANATIONS[pointer] = attrs_gri2[0].detach().numpy()
EXPLANATION_NAMES['IGrad'] = pointer
pointer += 1
# show_text_attr(attrs_gri2[0])


# In[25]:

# Evaluation metric 1 : Pearson Correlation
import numpy as np
corr_matrix = np.zeros([pointer, pointer])
for col_a, col_b in itertools.combinations_with_replacement(range(pointer), 2):
    corr_matrix[col_a][col_b], _ = pearsonr(EXPLANATIONS[col_a], EXPLANATIONS[col_b])
    corr_matrix[col_b][col_a] = corr_matrix[col_a][col_b]

# Evaluation metric 2 : Rank Correlation -- TODO


# In[26]: Plotting methods


def plot_confusion_matrix(index = None, column = None, data = None, plot_path = None):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    # array = [[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3],
    #          [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #          [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
    #          [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
    #          [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
    #          [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
    #          [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
    #          [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
    #          [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
    #          [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
    #          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]]
    df_cm = pd.DataFrame(data, index=index,
                         columns=column)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm) # annot=True)
    plt.savefig(plot_path)


# In[27]:


plot_confusion_matrix(index = EXPLANATION_NAMES.keys(), column = EXPLANATION_NAMES.keys(), data = corr_matrix, plot_path ="result_text_1.png")


# In[70]:


corr_matrix


# In[ ]:









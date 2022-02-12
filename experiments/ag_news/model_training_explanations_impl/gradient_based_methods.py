import torch
import captum
# from train_without_embedding_bag import TextClassificationModel
from tc_lstm import LSTMClassifier
from torchtext.datasets import AG_NEWS
from torch import nn
import pickle
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from captum.attr import LimeBase, KernelShap,InputXGradient
from captum._utils.models.linear_model import SkLearnLasso
import torch.nn.functional as F
# from IPython.core.display import HTML, display
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer, GradientShap
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization, IntegratedGradients
# import shap
from scipy.stats.stats import pearsonr
import itertools
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EXPLANATIONS = 3
EXPLANATIONS = dict()
EXPLANATION_NAMES = dict()
pointer = 0

def num_to_text(text_nums, vocab) :
    return [vocab.vocab.itos_[i] for i in text_nums]


PATH = "./text_classification_lstm.model"
tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(x)
label_pipeline = lambda x: int(x) - 1

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
model = torch.load(PATH, map_location=device)
import time


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

# Hyperparameters
EPOCHS = 1  # epoch
LR = 5  # learning rate
BATCH_SIZE = 4  # batch size for training

_, test_iter = AG_NEWS()
test_dataset = to_map_style_dataset(test_iter)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_batch)
model = torch.load(PATH, map_location=device)
print("The accuracy of the trained models : {}".format(evaluate(test_dataloader)))




# Explanation Method 1 : Integrated Gradients
start_time = time.time()

def construct_whole_bert_embeddings(input_ids, ref_input_ids, token_type_ids=None, ref_token_type_ids=None, position_ids=None, ref_position_ids=None):
        input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids).to(device)
        ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids).to(device)
        return input_embeddings, ref_input_embeddings

from captum.attr import IntegratedGradients, TokenReferenceBase, visualization

token_reference = TokenReferenceBase(reference_token_idx=0)
vis_data_records_ig = []

def interpret_sentence(model, test_text, test_labels, interpretable_embedding = None):
        model.zero_grad()
        # input_indices dim: [sequence_length]
        seq_length = test_text.shape[1]
        # generate reference indices for each sample

        reference_indices = token_reference.generate_reference(seq_length, device=device).repeat(test_text.shape[0], 1)
        input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(test_text, reference_indices, interpretable_embedding)
        # compute attributions and approximation delta using layer integrated gradients
        ig = IntegratedGradients(model)
        attributions_ig = ig.attribute(input_embeddings, ref_input_embeddings,
                                               n_steps=500, target=test_labels).to(device)
        return attributions_ig

model = torch.load(PATH, map_location=device)
model2 = torch.load(PATH, map_location=device) # For computing predictions since model will transform to new embeddings

interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')

attrs_igs = []
with torch.no_grad():

    for idx, (label, text) in enumerate(test_dataloader):
        start_time = time.time()
        predicted_label = torch.argmax(model2(text), dim = -1)
        attrs_ig = interpret_sentence(model, text, predicted_label,
                                      interpretable_embedding=interpretable_embedding)
        attrs_ig2 = torch.sum(attrs_ig, dim=-1)
        end_time = time.time()
        attrs_igs.extend(list(zip([" ".join(vocab.lookup_tokens(list(i))) for i in text], attrs_ig2)))
        print(len(attrs_igs), end_time - start_time)

with open(f'ig_explanation.pkl', 'wb') as fp:
    pickle.dump(attrs_igs, fp)

end_time= time.time()

print("Time taken for IG  : {}".format(end_time - start_time))

# Explanation Method 2 : Smooth Grad


from captum.attr import NoiseTunnel
from captum.attr import Saliency

start_time = time.time()


token_reference = TokenReferenceBase(reference_token_idx=0)
vis_data_records_ig = []


def interpret_sentence(model, test_text, test_labels, interpretable_embedding=None, num_samples=500):
    # model.eval()
    model.train()
    # generate reference indices for each sample

    noise_tunnel = NoiseTunnel(Saliency(model))
    input_embeddings = interpretable_embedding.indices_to_embeddings(test_text).to(device)
    attribution = noise_tunnel.attribute(input_embeddings,
                                         nt_type='smoothgrad',
                                         target=test_labels,
                                         nt_samples=num_samples).to(device)
    return attribution  # *input_embeddings

model = torch.load(PATH, map_location=device)
model2 = torch.load(PATH, map_location=device)
interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')

attrs_sgs = []
with torch.no_grad():

    for idx, (label, text) in enumerate(test_dataloader):
        start_time = time.time()
        predicted_label = torch.argmax(model2(text), dim = -1)
        attrs_smg = interpret_sentence(model, text, predicted_label,
                                       interpretable_embedding=interpretable_embedding, num_samples=500)
        attrs_smg2 = torch.sum(attrs_smg, dim=-1).to("cpu")
        end_time = time.time()
        attrs_sgs.extend(list(zip([" ".join(vocab.lookup_tokens(list(i))) for i in text], attrs_smg2)))
        print(len(attrs_sgs), end_time - start_time)

with open(f'sg_explanation.pkl', 'wb') as fp:
    pickle.dump(attrs_sgs, fp)


end_time= time.time()

print("Time taken for Smooth Grad  : {}".format(end_time - start_time))


# Explanation Method 3 : Grad
start_time = time.time()
token_reference = TokenReferenceBase(reference_token_idx=0)
vis_data_records_ig = []


def interpret_sentence(model, test_text, test_labels,interpretable_embedding=None):
    model.train()

    saliency_map = Saliency(model)
    input_embeddings = interpretable_embedding.indices_to_embeddings(test_text).to(device)
    attribution = saliency_map.attribute(input_embeddings, target=test_labels).to(device)
    return attribution  # *input_embeddings

model = torch.load(PATH, map_location=device)
model2 = torch.load(PATH, map_location=device)

interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')
attrs_gs = []
with torch.no_grad():

    for idx, (label, text) in enumerate(test_dataloader):
        start_time = time.time()
        predicted_label = torch.argmax(model2(text), dim = -1)
        attrs_gr = interpret_sentence(model, text, predicted_label,
                                      interpretable_embedding=interpretable_embedding)
        attrs_gr2 = torch.sum(attrs_gr, dim=-1).to("cpu")
        end_time = time.time()
        attrs_gs.extend(list(zip([" ".join(vocab.lookup_tokens(list(i))) for i in text], attrs_gr2)))
        print(len(attrs_gs), end_time - start_time)

with open(f'g_explanation.pkl', 'wb') as fp:
    pickle.dump(attrs_gs, fp)


end_time= time.time()

print("Time taken for Gradients  : {}".format(end_time - start_time))

# Explanation Method 4 : Gradients * Inputs
start_time = time.time()
token_reference = TokenReferenceBase(reference_token_idx=0)
vis_data_records_ig = []


def interpret_sentence(model, test_text, test_labels, interpretable_embedding=None):
    model.train()

    saliency_map = InputXGradient(model)
    input_embeddings = interpretable_embedding.indices_to_embeddings(test_text).to(device)
    print(input_embeddings.shape)
    attribution = saliency_map.attribute(input_embeddings, target=test_labels).to(device)
    return attribution  # *input_embeddings

model = torch.load(PATH, map_location=device)
model2 = torch.load(PATH, map_location=device)


interpretable_embedding = configure_interpretable_embedding_layer(model, 'word_embeddings')

attrs_igs = []
with torch.no_grad():

    for idx, (label, text) in enumerate(test_dataloader):
        start_time = time.time()
        predicted_label = torch.argmax(model2(text), dim = -1)
        attrs_gri = interpret_sentence(model, text, predicted_label, interpretable_embedding=interpretable_embedding)
        attrs_gri2 = torch.sum(attrs_gri, dim=-1).to("cpu")
        end_time = time.time()
        attrs_igs.extend(list(zip([" ".join(vocab.lookup_tokens(list(i))) for i in text], attrs_gri2)))
        print(len(attrs_igs), end_time - start_time)

with open(f'gi_explanation.pkl', 'wb') as fp:
    pickle.dump(attrs_igs, fp)








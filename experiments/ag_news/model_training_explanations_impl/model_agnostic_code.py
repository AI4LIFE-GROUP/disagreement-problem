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
    # print(total_count)
    return total_acc / total_count

# Hyperparameters
EPOCHS = 1  # epoch
LR = 5  # learning rate
BATCH_SIZE = 1 # batch size for training

_, test_iter = AG_NEWS()
test_dataset = to_map_style_dataset(test_iter)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_batch)
model = torch.load(PATH, map_location=device)



print("The accuracy of the trained models : {}".format(evaluate(test_dataloader)))



# Explanation Method : 1 : Lime Code
# remove the batch dimension for the embedding-bag model
model = torch.load(PATH, map_location=device)

def forward_func(text):
    return model(text).to(device)
#
# # encode text indices into latent representations & calculate cosine similarity
def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
    original_emb = torch.mean(model.word_embeddings(original_inp), dim = -2).to(device)
    perturbed_emb = torch.mean(model.word_embeddings(perturbed_inp), dim = -2).to(device)
    distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1).to(device)
    return torch.exp(-1 * (distance ** 2) / 2).to(device)
#
# # binary vector where each word is selected independently and uniformly at random
def bernoulli_perturb(text, **kwargs):
    probs = torch.ones_like(text) * 0.5
    return torch.bernoulli(probs).long().to(device)
#

def interp_to_input(interp_sample, original_input, **kwargs):
    return original_input[interp_sample.bool()].view(original_input.size(0), -1).to(device) #.to(device) #

lasso_lime_base = LimeBase(
    forward_func,
    interpretable_model=SkLearnLasso(alpha=0.08),
    similarity_func=exp_embedding_cosine_distance,
    perturb_func=bernoulli_perturb,
    perturb_interpretable_space=True,
    from_interp_rep_transform=interp_to_input,
    to_interp_rep_transform=None
)
#

model = torch.load(PATH, map_location=device)
attrs_lasos = []
with torch.no_grad():
    start_time = time.time()
    for idx, (label, text) in enumerate(test_dataloader):

        predicted_label = torch.argmax(model(text), dim = -1)
        attrs_laso = lasso_lime_base.attribute(
                        text,  # add batch dimension for Captum
                        target=predicted_label,
                        n_samples=500,
                        show_progress=False
                    ).to(device)
        attrs_lasos.extend(list(zip([" ".join(vocab.lookup_tokens(list(i))) for i in text], attrs_laso)))
        if len(attrs_lasos)%32 == 0:
            end_time = time.time()
            print(len(attrs_lasos), end_time - start_time)
            start_time = time.time()



with open(f'lime_explanation.pkl', 'wb') as fp:
    pickle.dump(attrs_lasos, fp)

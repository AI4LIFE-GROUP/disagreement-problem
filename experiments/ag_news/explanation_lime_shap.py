import torch
import captum
from main import TextClassificationModel
from torchtext.datasets import AG_NEWS
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from captum.attr import LimeBase,GradientShap, LayerGradientShap, KernelShap
from captum._utils.models.linear_model import SkLearnLasso
import torch.nn.functional as F
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import transformers

#from IPython.core.display import HTML, display
# import shap
import shap

def get_lime_attributions():

    pass

def get_shap_attributions():
    pass

def num_to_text(text_nums, vocab) :
    return [vocab.vocab.itos_[i] for i in text_nums]


def main() :
    PATH = "./text_classification.model"
    tokenizer = get_tokenizer('basic_english')
    train_iter = AG_NEWS(split='train')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x[0]))
    label_pipeline = lambda x: int(x) - 1

    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)



    train_iter = AG_NEWS(split='train')
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 64
    #model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    model = torch.load("text_classification.model")
    import time

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
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

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                #print("One test label : ", text.shape, offsets.shape)
                predicted_label = model(text, offsets)
                text_converted = num_to_text(text, vocab)
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
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])


    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, collate_fn=collate_batch)

    # Lime Code
    # remove the batch dimension for the embedding-bag model
    def forward_func(text, offsets):
        return model(text.squeeze(0), offsets)

    # encode text indices into latent representations & calculate cosine similarity
    def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
        original_emb = model.embedding(original_inp, None)
        perturbed_emb = model.embedding(perturbed_inp, None)
        distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
        return torch.exp(-1 * (distance ** 2) / 2)

    # binary vector where each word is selected independently and uniformly at random
    def bernoulli_perturb(text, **kwargs):
        probs = torch.ones_like(text) * 0.5
        return torch.bernoulli(probs).long()

    # remove absenst token based on the intepretable representation sample
    def interp_to_input(interp_sample, original_input, **kwargs):
        return original_input[interp_sample.bool()].view(original_input.size(0), -1)

    def get_lime_explanations(test_text, test_labels, test_offsets):
        lasso_lime_base = LimeBase(
            forward_func,
            interpretable_model=SkLearnLasso(alpha=0.08),
            similarity_func=exp_embedding_cosine_distance,
            perturb_func=bernoulli_perturb,
            perturb_interpretable_space=True,
            from_interp_rep_transform=interp_to_input,
            to_interp_rep_transform=None
        )
        return lasso_lime_base.attribute(
            test_text.unsqueeze(0),  # add batch dimension for Captum
            target=test_labels,
            additional_forward_args=(test_offsets,),
            n_samples=32000,
            show_progress=True
        ).squeeze(0)


    #Shap code for text classification


    def construct_whole_bert_embeddings(input_ids, ref_input_ids, test_offsets, interpretable_embedding,\
                                        token_type_ids=None, ref_token_type_ids=None, \
                                        position_ids=None, ref_position_ids=None):
        input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids, test_offsets)
        ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids, test_offsets)
        return input_embeddings, ref_input_embeddings


    def get_shap_explanations(test_text, test_labels, test_offsets, interpretable_embedding):
        baselines = torch.zeros(test_text.shape[0]).to(torch.int64)
        input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(test_text, baselines)

        layer_grad_shap = KernelShap(model)
        shap_attribution = layer_grad_shap.attribute(input_embeddings, ref_input_embeddings,method='gausslegendre',
                                                     additional_forward_args=(test_offsets,),
                                                     target=1)
        return shap_attribution








    ## Test Sample and Visualization
    test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
    test_line = ['US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
                 '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
                 'And it is too early to tell if he will match Aleksandr Dityatin, '
                 'the Soviet gymnast who won eight total medals in 1980.']

    test_labels, test_text, test_offsets = collate_batch([(test_label, test_line)])

    probs = F.softmax(model(test_text, test_offsets), dim=1).squeeze(0)
    print('Prediction probability:', round(probs[test_labels[0]].item(), 4))



    # test_labels, test_text, test_offsets = collate_batch([(test_label, test_line)])
    # lime_attribution = get_lime_explanations(test_text, test_labels, test_offsets)
    #
    # interpretable_embedding = configure_interpretable_embedding_layer(model, 'embedding')
    # shap_attribution = get_shap_explanations(test_text, test_labels, test_offsets, interpretable_embedding)

    # Integrated gradients

    from captum.attr import IntegratedGradients, TokenReferenceBase, visualization

    token_reference = TokenReferenceBase(reference_token_idx=0)
    vis_data_records_ig = []



    def interpret_sentence(model, test_text, test_offsets, test_labels,  pred, pred_ind, min_len=7, interpretable_embedding = None, label=0):
        model.zero_grad()
        # input_indices dim: [sequence_length]
        seq_length = test_text.shape[0]
        # predic

        # generate reference indices for each sample
        interpretable_embedding = configure_interpretable_embedding_layer(model, 'embedding')
        ig = IntegratedGradients(model)
        reference_indices = token_reference.generate_reference(seq_length, device=device)
        input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(test_text, reference_indices, test_offsets, interpretable_embedding)
        # compute attributions and approximation delta using layer integrated gradients

        attributions_ig = ig.attribute(input_embeddings, ref_input_embeddings, #additional_forward_args=(test_offsets,),\
                                               n_steps=500, return_convergence_delta=True, target=test_labels)
        return attributions_ig[0] # to get tensor
    #     add_attributions_to_visualizer(attributions_ig, test_text, pred, pred_ind, label, vis_data_records_ig)
    #
    #
    # def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, vis_data_records):
    #     attributions = attributions.sum(dim=2).squeeze(0)
    #     attributions = attributions / torch.norm(attributions)
    #     attributions = attributions.cpu().detach().numpy()
    #
    #     # storing couple samples in an array for visualization purposes
    #     vis_data_records.append(visualization.VisualizationDataRecord(
    #         attributions,
    #         pred,
    #         attributions.sum(),
    #         text))

    model = torch.load("text_classification.model")
    print(model)
    test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
    test_line = ['US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
                 '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
                 'And it is too early to tell if he will match Aleksandr Dityatin, '
                 'the Soviet gymnast who won eight total medals in 1980.']
    test_labels, test_text, test_offsets = collate_batch([(test_label, test_line)])
    pred = F.softmax(model(test_text, test_offsets), dim=1)
    pred_ind = torch.round(pred)
    #interpretable_embedding = configure_interpretable_embedding_layer(model, 'embedding')
    a = interpret_sentence(model, test_text, test_offsets, test_labels,  pred,pred_ind,  label=2)
    print(a)





if __name__ == "__main__":
    main()
    print("Code run successful")
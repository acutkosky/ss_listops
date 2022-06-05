
import torch
import os
import torchtext
from datasets import load_dataset


# not sure we need to do this, but the original lra code renames ] to X because
# their tokenizer "removes non-alphanumerics". Idk why that would be, nor do
# I know if pytorch's setup does anything similear (I don't think it does), but
# let's assume so.
# I also don't know why that means we don't need to rename [ also, so I'm going
# to do that too.
# also we remove the ( and ). I dunno why they were there in the first place...
def listops_tokenizer(s):
    return s.translate({ord("["): ord("Y"), ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()



class ListOpsDataset:
    def __init__(self, dataset, vocab, batch_size):
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = batch_size

    def get_collate_fn(self):

        def collate_fn(batch):
            input_ids = [example['input_ids'] for example in batch]
            targets = torch.tensor([example['Target'] for example in batch])
            lengths = torch.tensor([len(x) for x in input_ids])
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                            batch_first=True,
                                                            padding_value=self.vocab["pad"])
            return padded_input_ids, targets, lengths

        return collate_fn

    def get_dataloader(self, key):
        self.dataset.set_format(type='torch', columns=['input_ids', 'Target'])
        collate_fn = self.get_collate_fn()
        return torch.utils.data.DataLoader(self.dataset[key], batch_size=self.batch_size, collate_fn=collate_fn)


def listops_dataloader(data_dir=None, batch_size=32, append_bos=False, append_eos=True, l_max=2048, n_workers=4, cache=True):

    print("loading listops data...")

    if data_dir is None:
        # default to the environment variable DATA_DIR
        data_dir = os.getenv('DATA_DIR')

    listops_dataset = dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_dir, "basic_train.tsv"),
                "val": os.path.join(data_dir, "basic_val.tsv"),
                "test": os.path.join(data_dir, "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

    l_max = l_max - int(append_bos) - int(append_eos)

    print("tokenizing...")
    tokenize = lambda example: {"tokens": listops_tokenizer(example["Source"])[:l_max]}
    dataset = dataset.map(
        tokenize,
        remove_columns=["Source"],
        keep_in_memory=True,
        load_from_cache_file=cache,
        num_proc=max(n_workers, 1),
    )

    print("building vocab...")
    vocab = torchtext.vocab.build_vocab_from_iterator(
        dataset["train"]["tokens"],
        specials=(
            ["<pad>", "<unk>"]
            + (["<bos>"] if append_bos else [])
            + (["<eos>"] if append_eos else [])
        ),
    )
    vocab.set_default_index(vocab["<unk>"])

    print("applying vocab...")
    numericalize = lambda example: {
        "input_ids": vocab(
            (["<bos>"] if append_bos else [])
            + example["tokens"]
            + (["<eos>"] if append_eos else [])
        )
    }
    dataset = dataset.map(
        numericalize,
        remove_columns=["tokens"],
        keep_in_memory=True,
        load_from_cache_file=cache,
        num_proc=max(n_workers, 1),
    )

    print("done!")
    return ListOpsDataset(dataset, vocab, batch_size)






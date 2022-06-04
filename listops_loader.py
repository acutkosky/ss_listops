
import torch
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


def listops_dataloader(data_dir, batch_size=32, append_bos=False, append_eos=True, n_workers=4):
    listops_dataset = dataset = load_dataset(
            "csv",
            data_files={
                "train": str(data_dir / "basic_train.tsv"),
                "val": str(data_dir / "basic_val.tsv"),
                "test": str(data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

    l_max = l_max - int(append_bos) - int(append_eos)

    tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}
    dataset = dataset.map(
        tokenize,
        remove_columns=["Source"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(n_workers, 1),
    )
    vocab = torchtext.vocab.build_vocab_from_iterator(
        dataset["train"]["tokens"],
        specials=(
            ["<pad>", "<unk>"]
            + (["<bos>"] if append_bos else [])
            + (["<eos>"] if append_eos else [])
        ),
    )
    vocab.set_default_index(vocab["<unk>"])

    numericalize = lambda example: {
        "input_ids": vocab(
            (["<bos>"] if self.append_bos else [])
            + example["tokens"]
            + (["<eos>"] if self.append_eos else [])
        )
    }
    dataset = dataset.map(
        numericalize,
        remove_columns=["tokens"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(n_workers, 1),
    )

    def collate_fn(batch):
        input_ids = [example['input_ids'] for example in batch]
        targets = torch.tensor([example['Target'] for example in batch])
        lengths = torch.tensor([len(x) for x in input_ids])
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                           batch_first=True, 
                                                           padding_value=vocab["pad"])

        return input_ids, targets, lengths


    dataset.set_format(type='torch', columns=['input_ids', 'Target'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        
    return dataloader





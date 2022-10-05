import torch
from torch.utils.data import IterableDataset
from ga_ner.models import Document, Vocabulary
from typing import Iterator, TYPE_CHECKING
import pytorch_lightning as pl

if TYPE_CHECKING:
    from io import TextIOWrapper
# Create dateset class using sampling of Document class using window size.
# If length of document is less than window size, then pad the document with [PAD] idx.


class DocumentDataset(IterableDataset):
    def __init__(
        self,
        file_path,
        window_size,
        vocab: Vocabulary = None,
        type_text="iob",
        encoding="utf-8",
        random_state=42,
    ):
        """_summary_

        Args:
            file_path (_type_): _description_
            window_size (_type_): _description_
            vocab (Vocabulary, optional): If it is not None, then use this vocabulary of spacy
            type_text (str, optional): _description_. Defaults to "iob".
            encoding (str, optional): _description_. Defaults to 'utf-8'.
            random_state (int, optional): _description_. Defaults to 42.
        """
        self.file_path = file_path
        self.window_size = window_size
        self.type_text = type_text
        pl.seed_everything(random_state)
        torch.manual_seed(random_state)
        torch.use_deterministic_algorithms(True)
        self.encoding = encoding
        self.vocab = vocab
        self.mask_id = vocab.mask_id

    def parse_iob_file(self, file: "TextIOWrapper"):
        for sentence in file.read().split("\n\n"):
            doc = Document.from_iob(sentence, add_entities=False, vocab=self.vocab)
            yield doc

    def parse_file(self):
        with open(self.file_path, "r", encoding=self.encoding) as file:
            doc_iter: Iterator[Document] = []
            if self.type_text == "iob":
                doc_iter = self.parse_iob_file(file)
            else:
                raise ValueError("Not implemented yet.")
            for doc in doc_iter:
                self.mask_id = doc.mask_id
                targets = torch.zeros(doc.vocab_size).bool()
                num_targets = 0
                for sample in doc.sampling(self.window_size):
                    target = sample[-1]
                    targets[target] = True
                    num_targets += 1
                    if target == doc.sep_id:
                        yield sample, targets
                        num_targets = 0
                        targets = torch.zeros(doc.vocab_size).bool()
                    elif num_targets == 3:
                        yield sample, targets
                        num_targets = 0
                        targets = torch.zeros(doc.vocab_size).bool()

    def __iter__(self):
        for sample, targets in self.parse_file():

            # Select random index to mask.
            # mask_idx = torch.randint(0, len(sample), (1,)).item()
            # mask_idx = sample_id % len(sample)
            # Complete padding of the sample.
            if len(sample) < self.window_size:
                sample = sample + (0,) * (self.window_size - len(sample))

            input_ids = torch.tensor(sample)
            input_ids[-1] = self.mask_id
            yield input_ids, targets

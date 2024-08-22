"""Use convolutions over a sequence to efficiently count query sub-sequences"""

from torch.nn.functional import conv1d, pad
from einops import rearrange, repeat
from typing import List
import torch


class PatternBuffer:
    def __init__(
        self,
        query_strings: List[str],
        embedding: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        """Efficiently count the presence of `query_strings` in unseen strings
        after `embedding`.

        Args:
            query_strings (List[str]): List of query strings to count occurances of
            embedding (torch.Tensor): Tensor describing mapping between character ordinals and tensors
            device (torch.device, optional): Device to perform calculations on. Defaults to torch.device("cpu").
        """
        self.device = device
        self.query_strings = query_strings
        self.embedding = embedding.to(self.device)
        self.process_queries()

    def process_queries(self):
        """Embed `query_strings` with provided `embedding` and prepare `support` tensor
        for correcting counts from novel sequences.
        """
        # Process the query strings, and extract lengths for padding/support
        query_lengths = torch.Tensor([len(s) for s in self.query_strings]).int()
        self.longest_query = int(query_lengths.max().item())
        self.kernel_length = self.longest_query + (self.longest_query & 1)
        self.embedded_queries = embed_strings(
            self.query_strings, self.embedding, pad_str="N", pad_even=True
        ).to(self.device)

        # Generate support matrix to correct count errors caused by padding
        support_base = repeat(
            torch.arange(self.kernel_length), "l -> k l", k=len(self.query_strings)
        ).to(self.device)
        expanded_lengths = repeat(query_lengths, "k -> k l", l=support_base.shape[-1])
        self.support = support_base.clone().to(self.device)
        self.support[support_base > (self.kernel_length - expanded_lengths)] = 0.0

    def count(self, string_list: List[str]):
        """Count occurances of provided `query_strings` in new `string_list`

        Args:
            string_list (List[str]): List of strings to be embedded and have counts taken

        Returns:
            torch.TensorType: Tensor of counts, arranged (seq x query)
        """
        # Embed input sequences and pad to allow full-length counting
        embedded_seqs = embed_strings(string_list, self.embedding, pad_str="Z")
        embedded_seqs = embedded_seqs.to(self.device)
        padded_seqs = pad(embedded_seqs, (0, self.longest_query), "constant", 0.0)

        # Count subsequence occurance with corrected 1d convolution
        convolution_out = conv1d(padded_seqs, weight=self.embedded_queries)
        convolution_out[:, :, -self.kernel_length :] += self.support
        return torch.sum(convolution_out == self.kernel_length, dim=-1)


def generate_iupac_embedding():
    """Generate a tensor that maps DNA IUPAC characters to sparse categorical
    tensors that describe the possible base compositions.

    Returns:
        torch.TensorType: Embedding tensor
    """
    embed = torch.zeros(256, 4).float()
    embed[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
    embed[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
    embed[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
    embed[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
    embed[ord("R")] = torch.Tensor([1.0, 0.0, 1.0, 0.0])
    embed[ord("Y")] = torch.Tensor([0.0, 1.0, 0.0, 1.0])
    embed[ord("S")] = torch.Tensor([0.0, 1.0, 1.0, 0.0])
    embed[ord("W")] = torch.Tensor([1.0, 0.0, 0.0, 1.0])
    embed[ord("K")] = torch.Tensor([0.0, 0.0, 1.0, 1.0])
    embed[ord("M")] = torch.Tensor([1.0, 1.0, 0.0, 0.0])
    embed[ord("B")] = torch.Tensor([0.0, 1.0, 1.0, 1.0])
    embed[ord("D")] = torch.Tensor([1.0, 0.0, 1.0, 1.0])
    embed[ord("H")] = torch.Tensor([1.0, 1.0, 0.0, 1.0])
    embed[ord("V")] = torch.Tensor([1.0, 1.0, 1.0, 0.0])
    embed[ord("N")] = torch.Tensor([1.0, 1.0, 1.0, 1.0])
    embed[ord("Z")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
    return embed


def embed_strings(
    string_list: List[str],
    embedding: torch.Tensor,
    pad_str: str = None,
    pad_even: bool = False,
):
    """Embed all strings in a list into a categorical embedding using the given
    embedding tenosr

    Args:
        string_list (List[str]): List of strings to be embedded
        embedding (torch.Tensor): Tensor describing mapping between character ordinals and tensors
        pad_str (str): If provided, will pad all strings to same length using this character
        pad_even (bool): Should the padding length be rounded up to nearest even numbers?

    Returns:
        torch.TensorType: Tensor of categorical embeddings for each string
    """
    # Apply padding if requested
    if pad_str is not None:
        max_length = int(max([len(s) for s in string_list]))
        if pad_even:
            max_length += max_length & 1
        string_list = [s.ljust(max_length, pad_str) for s in string_list]
    # Embed each of the strings
    seq_tensors = torch.stack(
        [torch.stack([embedding[ord(s)] for s in seq]) for seq in string_list]
    )
    return rearrange(seq_tensors, "batch length channel -> batch channel length")


@torch.no_grad()
def count_queries(
    seqs: List[str],
    queries: List[str],
    embedding: torch.Tensor,
    device: torch.device = torch.device("cpu"),
):
    """Count number of occurance of each query in the seqs after each has been
    embedded. Can be done on the GPU for signficiant speedups.

    Args:
        seqs (List[str]): List of strings to be embedded and have counts taken
        queries (List[str]): List of query strings to count occurances of
        embedding (torch.Tensor): Tensor describing mapping between character ordinals and tensors
        device (torch.DeviceObjType, optional): Device to perform calculations on. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: Tensor of counts, arranged (seq x query)
    """
    # Process the query strings, and extract lengths for padding/support
    query_lengths = torch.Tensor([len(s) for s in queries]).int()
    longest_query = int(query_lengths.max().item())
    kernel_length = longest_query + (longest_query & 1)
    embedded_queries = embed_strings(queries, embedding, pad_str="N", pad_even=True)
    embedded_queries = embedded_queries.to(device)

    # Generate support matrix to correct count errors caused by padding
    support_base = repeat(torch.arange(kernel_length), "l -> k l", k=len(queries))
    expanded_lengths = repeat(query_lengths, "k -> k l", l=support_base.shape[-1])
    support = support_base.clone().to(device)
    support[support_base > (kernel_length - expanded_lengths)] = 0.0

    # Embed target sequences and pad to allow full-length counting
    embedded_seqs = embed_strings(seqs, embedding, pad_str="Z")
    padded_seqs = pad(embedded_seqs, (0, longest_query), "constant", 0.0).to(device)

    # Count subsequence occurance with corrected 1d convolution
    convolution_out = conv1d(padded_seqs, weight=embedded_queries)
    convolution_out[:, :, -kernel_length:] += support
    return torch.sum(convolution_out == kernel_length, dim=-1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = [
        "AACGAATCAAAAT",
        "AACAGTTCAAAAATTAGT",
        "AGTTCGYGGA",
        "AACAAGATCAGGAAAGCTGACTTGATG",
    ]
    query_seqs = ["AAA", "AGT", "AAACA", "AAR", "GYGGA"]
    embedding = generate_iupac_embedding()

    # Functional interface: process seqs and embedding together then count
    counts = count_queries(sequences, query_seqs, embedding, device)
    print(counts)

    # OO Interface: create with queries/embeddings; count on new sequences
    counter = PatternBuffer(query_strings=query_seqs, embedding=embedding)
    print(counter.count(sequences))

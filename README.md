# Pattern Buffer

Fast pattern counting on small-alphabet sequences with GPU acceleration, built on PyTorch's `Tensor`.

Originally designed to count patterns in DNA sequences with ambiguous bases as defined by the [IUPAC code](https://www.bioinformatics.org/sms/iupac.html), but can be extended to any pattern counting task where the original sequences and target patterns can be converted into a sensible categorical encoding. 

> [!NOTE]
> The preface of "small-alphabet" sequences is given as the additional memory used by categorical encoding grows with the number of unique values in the alphabet. With DNA and IUPAC encoding this is tractable because many of the codes represent combinations of DNA bases, so despite there being 15 DNA nucleotide codes we only need to embed in a vector of length 4 (A, C, G, T).

## Usage

> [!IMPORTANT]
> While the query sequences can all be different lengths, currently `Pattern Buffer` only supports counting these patterns on sequences that are all the same length. 

`Pattern Buffer` can be used with a broadly `functional` or `object-oriented (OO)` interface, with the `functional` interface geared towards one-time use and the `OO` interface for use multiple times (i.e. file parsing or PyTorch DataLoaders).

To demonstrate, we'll first create some sample sequences and queries. As we're using IUPAC nucleotide sequences, we can use the provided `generate_iupac_embedding` function to provide the embedding tensor. 

```python
>>> from pattern_buffer import generate_iupac_embedding

# Create sample sequences and queries
>>> sequences = ["AACGAATCAAAAT", "AACAGTTCAAAAT", "AACAGTTCGYGGA", "AACAAGATCAGGA"]
>>> queries = ["AAA", "AGT", "AAACA", "AAR", "GYGGA"]
>>> embedding = generate_iupac_embedding()
```

From here, we can use the function `count_queries` to count the queries in a single function
call: 
```python
>>> from pattern_buffer import count_queries
>>> count_queries(sequences, queries, embedding)
tensor([[2, 0, 0, 2, 0],
        [2, 1, 0, 2, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]])
```

or create an instance of the `PatternBuffer` class, and use the `.count` method to count occurrences in new sequences. This has the advantage of not re-calculating the query embeddings or the `support` tensor each time, so is well suited for fast repeated counting:

```python
>>> from pattern_buffer import PatternBuffer
>>> pb = PatternBuffer(query_strings=queries, embedding=embedding)
>>> pb.count(sequences)
tensor([[2, 0, 0, 2, 0],
        [2, 1, 0, 2, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]])
```

## Limitations
- Currently, the program expects all input sequences to have the same length, but queries can already be different lengths. 
- If all of your patterns contain unique (non-ambiguous) characters then this encoding scheme is likely overkill and other software would be more efficient. 
- The software is designed for use with GPU acceleration, and will likely under-perform on CPU when compared to CPU-optimised counting schemes.

## Future work 
- [ ] Allow dynamic input lengths with padding
- [ ] Automatic encoding detection from pattern analysis
- [ ] FFT-based convolutions for long patterns

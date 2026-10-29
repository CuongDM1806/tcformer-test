import numpy as np
import torch


def interaug(batch):
    x, y = batch
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    current = 0
    n_chunks = 9 if new_samples.shape[-1] == 1125 else (8 if new_samples.shape[-1] % 8 == 0 else 7) # special case for BCIC III
    for cls in torch.unique(y):
        x_cls = x[y == cls]
        chunks = torch.cat(torch.chunk(x_cls, chunks=n_chunks, dim=-1))
        # indices = np.random.choice(len(x_cls), size=(len(x_cls), n_chunks),
        #                            replace=True)
        indices = torch.randint(0, len(x_cls), size=(len(x_cls), n_chunks), device=x_cls.device)  

        for idx in indices:
            # add offset
            idx += np.arange(0, chunks.shape[0], len(x_cls))

            # create new sample
            new_sample = chunks[idx]
            new_sample = new_sample.permute(1, 0, 2).reshape(
                1, x_cls.shape[1], x_cls.shape[2])
            new_samples[current] = new_sample
            new_labels[current] = cls
            current += 1

    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)

    # shuffle
    perm = torch.randperm(len(combined_x))
    combined_x = combined_x[perm]
    combined_y = combined_y[perm]

    return combined_x, combined_y


def interaug_vectorized(batch):
    """Same segment recombination as ``interaug`` without the per-sample loop.

    Random numbers are drawn with the same calls in the same order (one
    ``randint`` per class, then ``randperm``), so under the same seed the
    output is identical to ``interaug``.
    """
    x, y = batch
    n_samples, n_channels, n_times = x.shape
    n_chunks = 9 if n_times == 1125 else (8 if n_times % 8 == 0 else 7) # special case for BCIC III
    segments = x.reshape(n_samples, n_channels, n_chunks, n_times // n_chunks)
    chunk_ids = torch.arange(n_chunks, device=x.device)
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    current = 0
    for cls in torch.unique(y):
        members = torch.nonzero(y == cls, as_tuple=True)[0]
        indices = torch.randint(0, len(members), size=(len(members), n_chunks), device=x.device)
        # Segment k of each new sample comes from segment k of a random same-class trial.
        picked = segments[members[indices], :, chunk_ids]  # [n_cls, n_chunks, C, T/n_chunks]
        new_samples[current:current + len(members)] = picked.permute(0, 2, 1, 3).reshape(
            len(members), n_channels, n_times)
        new_labels[current:current + len(members)] = cls
        current += len(members)

    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)

    # shuffle
    perm = torch.randperm(len(combined_x))
    combined_x = combined_x[perm]
    combined_y = combined_y[perm]

    return combined_x, combined_y

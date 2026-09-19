import torch


def make_interaugmented_batch(batch):
    """Create one same-size batch by recombining within-class time chunks."""
    x, y = batch
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    current = 0
    n_chunks = 9 if new_samples.shape[-1] == 1125 else (8 if new_samples.shape[-1] % 8 == 0 else 7) # special case for BCIC III
    for cls in torch.unique(y):
        x_cls = x[y == cls]
        chunks = torch.cat(torch.chunk(x_cls, chunks=n_chunks, dim=-1))
        indices = torch.randint(
            0, len(x_cls), size=(len(x_cls), n_chunks), device=x_cls.device
        )
        offsets = torch.arange(
            0, chunks.shape[0], len(x_cls), device=x_cls.device
        )

        for idx in indices:
            idx = idx + offsets

            # create new sample
            new_sample = chunks[idx]
            new_sample = new_sample.permute(1, 0, 2).reshape(
                1, x_cls.shape[1], x_cls.shape[2])
            new_samples[current] = new_sample
            new_labels[current] = cls
            current += 1

    perm = torch.randperm(len(new_samples), device=new_samples.device)
    return new_samples[perm], new_labels[perm]


def interaug(batch):
    """Return the legacy concatenated original + augmented source batch."""
    x, y = batch
    new_samples, new_labels = make_interaugmented_batch(batch)
    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)

    # shuffle
    perm = torch.randperm(len(combined_x), device=combined_x.device)
    combined_x = combined_x[perm]
    combined_y = combined_y[perm]

    return combined_x, combined_y

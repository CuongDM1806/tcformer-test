import torch


def interaug(batch):
    if len(batch) not in (2, 3):
        raise ValueError("InterAug expects [x, y] or [x, y, subject_id].")
    x, y = batch[:2]
    subject_ids = batch[2] if len(batch) == 3 else None
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    new_subject_ids = (
        torch.zeros_like(subject_ids) if subject_ids is not None else None
    )
    current = 0
    n_chunks = 9 if new_samples.shape[-1] == 1125 else (8 if new_samples.shape[-1] % 8 == 0 else 7) # special case for BCIC III
    if subject_ids is None:
        groups = [(y == cls, cls, None) for cls in torch.unique(y)]
    else:
        # A synthetic trial must have an unambiguous subject label. Therefore,
        # mix segments only within the same class and the same source subject.
        class_subject_pairs = torch.unique(
            torch.stack((y, subject_ids), dim=1), dim=0
        )
        groups = [
            (
                (y == pair[0]) & (subject_ids == pair[1]),
                pair[0],
                pair[1],
            )
            for pair in class_subject_pairs
        ]

    for group_mask, cls, subject_id in groups:
        x_cls = x[group_mask]
        chunks = torch.cat(torch.chunk(x_cls, chunks=n_chunks, dim=-1))
        # indices = np.random.choice(len(x_cls), size=(len(x_cls), n_chunks),
        #                            replace=True)
        indices = torch.randint(0, len(x_cls), size=(len(x_cls), n_chunks), device=x_cls.device)  

        for idx in indices:
            # add offset
            idx += torch.arange(
                0, chunks.shape[0], len(x_cls), device=idx.device
            )

            # create new sample
            new_sample = chunks[idx]
            new_sample = new_sample.permute(1, 0, 2).reshape(
                1, x_cls.shape[1], x_cls.shape[2])
            new_samples[current] = new_sample
            new_labels[current] = cls
            if new_subject_ids is not None:
                new_subject_ids[current] = subject_id
            current += 1

    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)
    if subject_ids is not None:
        combined_subject_ids = torch.cat((subject_ids, new_subject_ids), dim=0)

    # shuffle
    perm = torch.randperm(len(combined_x), device=combined_x.device)
    combined_x = combined_x[perm]
    combined_y = combined_y[perm]
    if subject_ids is not None:
        combined_subject_ids = combined_subject_ids[perm]
        return combined_x, combined_y, combined_subject_ids

    return combined_x, combined_y

# Zhou2016, recordings for SSM-DAN

Data-only branch of the SSM-DAN repository (the code is on `main`). Its files go to `data/`
on `main` unchanged: `MNE-zhou-2016/` here is `data/MNE-zhou-2016/` there. `python -m ssm_dan.launch`
and `notebooks/zhou2016.ipynb` fetch this branch when the files are not in `data/` yet.

`MNE-zhou-2016/subject_{1..4}/{session}{run}.cnt`: 4 subjects, sessions 1-3, runs A and B
(Neuroscan CNT, 24 files). The files are the original recordings, unchanged. They come from the archive
that MOABB 0.5 downloads (figshare file 3662952), whose `S{subject}_{session}{run}.cnt` files MOABB
renames to this layout.

Original data: B. Zhou, X. Wu, Z. Lv, L. Zhang, X. Guo, "A fully automated trial selection method for
optimization of motor imagery based brain-computer interface", PLoS ONE 11(9): e0162657, 2016.

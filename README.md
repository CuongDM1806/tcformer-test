# BCI Competition IV-2a, official files for SSM-DAN

Data-only branch of the SSM-DAN repository (the code is on `main`). Its folder `BCICIV_2a/` is
`data/BCICIV_2a/` on `main`; `python -m ssm_dan.launch` and `notebooks/bcic2a.ipynb` fetch it when
the files are not there yet, and build `data/BCICIV_2a/mat/` from them.

```
BCICIV_2a/gdf/           A01T.gdf ... A09E.gdf   BCICIV_2a_gdf.zip, unzipped
BCICIV_2a/true_labels/   A01T.mat ... A09E.mat   ds2a/true_labels.zip, unzipped
```

The files are the official ones of the BCI Competition IV, unchanged:
- https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
- https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip

Data set 2a of the BCI Competition IV (data set 001-2014 of BNCI Horizon 2020). Licensor: Institute
for Knowledge Discovery, Graz University of Technology; license CC BY-ND 4.0
(https://creativecommons.org/licenses/by-nd/4.0/). Reference: C. Brunner, R. Leeb, G. R. Müller-Putz,
A. Schlögl, G. Pfurtscheller, "BCI Competition 2008 - Graz data set A", 2008.

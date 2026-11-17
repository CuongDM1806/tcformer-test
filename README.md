# BCI Competition IV-2b, official files for SSM-DAN

Data-only branch of the SSM-DAN repository (the code is on `main`). Its folder `BCICIV_2b/` is
`data/BCICIV_2b/` on `main`; `python -m ssm_dan.launch` and `notebooks/bcic2b.ipynb` fetch it when
the files are not there yet, and build `data/BCICIV_2b/mat/` from them.

```
BCICIV_2b/gdf/           B0101T.gdf ... B0905E.gdf   BCICIV_2b_gdf.zip, unzipped
BCICIV_2b/true_labels/   B0101T.mat ... B0905E.mat   ds2b/true_labels.zip, unzipped
```

The files are the official ones of the BCI Competition IV, unchanged:
- https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip
- https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip

Data set 2b of the BCI Competition IV (data set 004-2014 of BNCI Horizon 2020). Licensor: Institute
for Knowledge Discovery, Graz University of Technology; license CC BY-ND 4.0
(https://creativecommons.org/licenses/by-nd/4.0/). Reference: R. Leeb, C. Brunner, G. R. Müller-Putz,
A. Schlögl, G. Pfurtscheller, "BCI Competition 2008 - Graz data set B", 2008.

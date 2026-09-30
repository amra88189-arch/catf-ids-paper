# Data

## 1. TON_IoT (download)

TON_IoT is published by UNSW Canberra and is free for academic research:
**https://research.unsw.edu.au/projects/toniot-datasets**

From the download, take the **Processed_datasets** folder.

Place it at `data/TON_IoT/Processed_datasets/`, or pass its location with `--base`.
The scripts read these 38 files:

```
Processed_datasets/
  Processed_Network_dataset/   Network_dataset_1.csv ... Network_dataset_23.csv
  Processed_IoT_dataset/       IoT_Fridge.csv, IoT_Garage_Door.csv, IoT_GPS_Tracker.csv,
                               IoT_Modbus.csv, IoT_Motion_Light.csv, IoT_Thermostat.csv,
                               IoT_Weather.csv
  Processed_Linux_dataset/     linux_disk_1.csv, linux_disk_2.csv, linux_memory1.csv,
                               linux_memory2.csv, Linux_process_1.csv, Linux_process_2.csv
  Processed_Windows_dataset/   windows7_dataset.csv, windows10_dataset.csv
```

## 2. Processed, fused dataset (not stored here)

Because of its size, the fused dataset the paper evaluates is not stored in the
repository. Rebuild it from the download with:

```
cd code
python export_processed_data.py --base "path/to/Processed_datasets"
```

It is written to `data/processed/`:

| File | Records | Attack | Normal |
|---|---|---|---|
| `catf_fused_stream.csv.gz` — the full fused stream, time order | 24,934 | 18,306 | 6,628 |
| `catf_fused_balanced.csv.gz` — the class-balanced training set | 13,256 | 6,628 | 6,628 |

The script checks both counts against the paper. `manifest.json` lists the column
groups (the 32 network, 7 device and 10 Linux model features; the per-source labels
and the fused label; row id and timestamp; other columns the models do not use),
SHA-256 checksums and package versions.

How the dataset is built:

- 2,000 records drawn from each network file (1,000 per class where available),
  and the same number per device, Linux and Windows file.
- Each source preprocessed into its feature vector (Section 4.2 of the paper).
- Joined with `pandas.merge_asof` on the timestamp: network records as the base,
  each matched to the nearest device, Linux and Windows record within 300 s.
  Records without all partners are dropped.
- Fused label by weighted vote (network 0.45, device 0.20, Linux 0.20,
  Windows 0.15); attack if the vote reaches 0.45.
- The balanced set is an equal-size random sample of each class
  (`random_state=42`), sorted by time.

The Windows columns are loaded and their label enters the vote, but the models do
not use Windows features.

## Terms of use and citation

TON_IoT is © UNSW Canberra. Free use for academic research is granted by its
authors; commercial use requires permission from Dr Nour Moustafa. The dataset's
authors ask users to cite:

1. N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: network TON_IoT datasets," *Sustainable Cities and Society*, vol. 72, art. 102994, 2021.
2. T. M. Booij, I. Chiscop, E. Meeuwissen, N. Moustafa and F. T. H. den Hartog, "ToN_IoT: the role of heterogeneity and the need for standardization of features and attack types in IoT network intrusion data sets," *IEEE Internet of Things Journal*, vol. 9, no. 1, pp. 485–496, 2022.
3. A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood and A. Anwar, "TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven intrusion detection systems," *IEEE Access*, vol. 8, pp. 165130–165150, 2020.
4. N. Moustafa, M. Keshk, E. Debie and H. Janicke, "Federated TON_IoT Windows datasets for evaluating AI-based security applications," in *Proc. IEEE TrustCom*, 2020, pp. 848–855.
5. N. Moustafa, M. Ahmed and S. Ahmed, "Data analytics-enabled intrusion detection: evaluations of ToN_IoT Linux datasets," in *Proc. IEEE TrustCom*, 2020, pp. 727–735.
6. N. Moustafa, "New generations of Internet of Things datasets for cybersecurity applications based machine learning: TON_IoT datasets," in *Proc. eResearch Australasia Conference*, Brisbane, 2019.
7. N. Moustafa, "A systemic IoT–fog–cloud architecture for big-data analytics and cyber security systems: a review of fog computing," arXiv:1906.01055, 2019.
8. J. Ashraf, M. Keshk, N. Moustafa, M. Abdel-Basset, H. Khurshid, A. D. Bakhshi and R. R. Mostafa, "IoTBoT-IDS: a novel statistical learning-enabled botnet detection framework for protecting networks of smart cities," *Sustainable Cities and Society*, vol. 72, art. 103041, 2021.

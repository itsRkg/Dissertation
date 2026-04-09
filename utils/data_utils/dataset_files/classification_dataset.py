import torch
from torch.utils.data import Dataset

class MultiStationRainfallDataset(Dataset):
    def __init__(self, df, context_length, horizon):
        self.context_length = context_length
        self.horizon = horizon
        self.seq_len = context_length + horizon

        self.samples = []
        skipped_stations = 0

        for station_id, group in df.groupby('station_id'):
            group = group.sort_values('date')

            tokens = group['token'].values
            doy = group['day_of_year'].values

            n = len(tokens)

            # Skip stations that are too short
            if n < self.seq_len:
                skipped_stations += 1
                continue

            for i in range(n - self.seq_len):
                self.samples.append((
                    tokens[i:i+self.seq_len],
                    doy[i:i+self.seq_len]
                ))

        print(f"Skipped {skipped_stations} stations with length < {self.seq_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_seq, doy_seq = self.samples[idx]

        return {
            "tokens": torch.tensor(token_seq, dtype=torch.long),
            "doy": torch.tensor(doy_seq, dtype=torch.long)
        }

class TransformedReader:
    def __init__(self, reader, transform):
        self.reader = reader
        self.transform = transform

    def __len__(self):
        return len(self.reader.files)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform:
            data = self.transform(data)
        return data

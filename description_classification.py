import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW
from tqdm.auto import tqdm

# Get Data into the following desired format
'''
data = {
    'description': ['They are dealing in software products',
                    'They are brewing and selling alcohol, especially beer',
                    'Company C manufacturing and selling medical equipment',
                    'Company is operating a chain of gambling casinos',
                    'Company E offering cloud computing and data analytics services',
                    'Company F producing and distributing pork products',
                    ],
    'sharia_compliant': [1, 0, 1, 0, 1, 0]  # 1 for compliant, 0 for non-compliant
}
'''
def prepare_training_data(data):
    data = {
        'description': list(data.keys()),
        'sharia_compliant': list(data.values())
    }
    df = pd.DataFrame(data)

    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.02)
    return train_df, val_df


'''
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df['description'], df['sharia_compliant'], test_size=0.1)

# Text vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Prediction
X_test_tfidf = vectorizer.transform(X_test)
predictions = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
'''


def evaluate_model(model, data_loader, device):
    model.eval()

    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            predictions.extend(preds.detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy



class CustomDataset(Dataset):
    def __init__(self, description, sharia_compliant, tokenizer, max_len):
        self.description = description
        self.sharia_compliant = sharia_compliant
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.description)

    def __getitem__(self, item):
        text = str(self.description[item])
        sharia_compliant = self.sharia_compliant[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sharia_compliant, dtype=torch.long)
        }


# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 512


def create_datasets(train_df, val_df, tokenizer, max_len):
    # Create datasets
    train_dataset = CustomDataset(
        description=train_df.description.to_numpy(),
        sharia_compliant=train_df.sharia_compliant.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    val_dataset = CustomDataset(
        description=val_df.description.to_numpy(),
        sharia_compliant=val_df.sharia_compliant.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )



    batch_size = 1

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    return train_data_loader, val_data_loader

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=2)  # Sharia and Non Sharia compliant


def train_model(model, train_data_loader, val_data_loader):

    optimizer = AdamW(model.parameters(), lr=2e-5)  # Learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epochs = 3

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_data_loader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            print(loss)

        val_accuracy = evaluate_model(model, val_data_loader, device)
        print(f"Validation Accuracy after epoch {epoch}: {val_accuracy}")

    return model



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)


# Accuracy Testing and Validation

def prepare_description_data(snp500):
    snp500 = {
        'description': list(snp500.keys()),
        'sharia_compliant': list(snp500.values())
    }
    df2 = pd.DataFrame(snp500)
    return df2

def create_description_datasets(df2, batch_size=1):
    val_dataset = CustomDataset(
        description=df2.description.to_numpy(),
        sharia_compliant=df2.sharia_compliant.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    val_data_loader2 = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    return val_data_loader2

def evaluate_model_testing(model, data_loader, device):
    model.eval()

    predictions, true_labels = [], []
    total_examples, correct_predictions = 0, 0

    with torch.no_grad():
        for batch in data_loader:
            descriptions = batch['text']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            predictions.extend(preds.detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

            for description, label, pred in zip(descriptions, labels, preds):
                print(f"Description: {description}\nTrue Label: {label.item()}, Predicted Label: {pred.item()}\n")

            total_examples += labels.size(0)
            correct_predictions += (preds == labels).sum().item()

    accuracy = correct_predictions / total_examples
    return accuracy


def evaluate_model_testing_with_count(model, data_loader, device):
    model.eval()

    predictions, true_labels = [], []
    zero_classifications_count = []  # Array to hold counts for classifications of 0
    total_examples, correct_predictions, current_count = 0, 0, 0

    with torch.no_grad():
        for batch in data_loader:
            descriptions = batch['text']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            predictions.extend(preds.detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

            for description, label, pred in zip(descriptions, labels, preds):
                current_count += 1  # Increment the count for each item processed
                if pred.item() == 0:  # Check if the prediction is 0
                    zero_classifications_count.append(current_count - 1)  # Append the current count
                print(f"Description: {description}\nTrue Label: {label.item()}, Predicted Label: {pred.item()}\n")

            total_examples += labels.size(0)
            correct_predictions += (preds == labels).sum().item()

    accuracy = correct_predictions / total_examples
    print(f"Validation Accuracy: {accuracy}")
    return zero_classifications_count








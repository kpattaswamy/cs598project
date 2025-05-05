import torch
from constants import NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, PAD_IDX


def collate_fn_labels(batch):
    # batch is a list of dicts with keys: 'imp', 'label', 'len'
    input_seqs = [b["imp"] for b in batch]
    labels = [b["label"] for b in batch]
    lengths = [b["len"] for b in batch]

    padded_input = torch.nn.utils.rnn.pad_sequence(
        input_seqs, batch_first=True, padding_value=PAD_IDX
    )
    labels_tensor = torch.stack(labels)

    return {"imp": padded_input, "label": labels_tensor, "len": lengths}

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split

    from datasets.impressions_dataset import ImpressionsDataset
    from models.bert_labeler import bert_labeler
    from utils import evaluate, generate_attention_masks

    from constants import NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, PAD_IDX
    import torch

    # === Step 1: Load and Split Dataset ===
    print("📦 Loading dataset...")
    dataset_path = "../../clean_dataset.csv"  # Adjust if needed
    tokenized_path = "encoded_impressions.json"

    full_dataset = ImpressionsDataset(dataset_path, tokenized_path)

    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    # train_loader = DataLoader(train_ds, batch_size=18, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_ds, batch_size=18, shuffle=False, num_workers=0)

    # train_loader = DataLoader(train_ds, batch_size=18, shuffle=True, num_workers=0, collate_fn=collate_fn_labels)
    # val_loader = DataLoader(val_ds, batch_size=18, shuffle=False, num_workers=0, collate_fn=collate_fn_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn_labels)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn_labels)


    # === Step 2: Initialize Model ===
    print("🧠 Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = bert_labeler()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    # === Step 3: Train Model ===
    print("🚀 Starting training...")
    num_epochs = 8

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["imp"].to(device)
            labels = batch["label"].to(device).permute(1, 0)  # shape: (14, batch_size)
            lengths = batch["len"]
            attn_mask = generate_attention_masks(input_ids, lengths, device)

            outputs = model(input_ids, attn_mask)

            # Compute average loss across 14 conditions
            loss = sum(criterion(outputs[i], labels[i]) for i in range(14)) / 14

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

        # === Step 4: Evaluate ===
        model.eval()
        metrics = evaluate(model, val_loader, device)
        print(f"📊 AUCs: {metrics['auc']}\n")

    # === Step 5: Save Model ===
    torch.save({'model_state_dict': model.state_dict()}, 'my_visualchexbert_model.pth')
    print("💾 Model saved as my_visualchexbert_model.pth")


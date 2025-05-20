from tqdm import tqdm


def trainModel(model, epochs, train_loader, val_loader):

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.sigmoid(outputs) > 0.5
        acc = (pred == masks).float().mean()

        train_loss += loss.item()
        train_acc += acc.item()

        loop.set_postfix({
            "loss": loss.item(),
            "acc": acc.item()
        })

    # Validation phase
    model.eval()
    val_loss = 0
    dice_score = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Calculate Dice coefficient
            pred = torch.sigmoid(outputs) > 0.5
            dice_score += (2 * (pred * masks).sum()) / ((pred + masks).sum() + 1e-8)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    # Print metrics
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, Dice: {dice_score/len(val_loader):.4f}')

    # Update learning rate based on validation losp
    scheduler.step(val_loss)

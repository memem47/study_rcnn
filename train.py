criterion_bce = nn.BCEWithLogitsLoss()
criterion_dice = lambda p, g: 1 - (2 * (p*g).sum() + 1) / (p.pow(2).sum() + g.pow(2).sum() + 1)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

for epoch in range(epochs):
    for x, y_true in loader:           # x:(B,T,C,H,W), y_true:(B,T,1,H,W)
        y_pred = model(x)
        loss = criterion_bce(y_pred, y_true) + 0.5 * criterion_dice(torch.sigmoid(y_pred), y_true)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()

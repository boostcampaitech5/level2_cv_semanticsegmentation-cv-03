import datetime
import os

from tqdm import tqdm
import wandb

import torch
import torch.nn.functional as F

from metric import dice_coef


def train(
    model,
    data_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    classes,
    patience,
    save_dir,
):
    print(f"Start training..")
    best_dice = 0.0
    best_epoch = 0
    check_patience = 0

    for epoch in range(epochs):
        model.train()

        for step, (images, masks) in enumerate(data_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            # inference
            outputs = model(images)['out']

            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Step [{step+1}/{len(data_loader)}], "
                    f"Loss: {round(loss.item(),4)}"
                )

            wandb.log({"Train loss": loss.item()})

        # validation 주기에 따른 loss 출력 및 best model 저장
        dice = validation(epoch + 1, model, val_loader, criterion, classes)

        if best_dice < dice:
            print(
                f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}"
            )

            best_dice = dice
            best_epoch = epoch + 1
            check_patience = 0

            # Save best model
            output_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model, output_path)
        else:
            check_patience += 1

        wandb.log({"VALID DICE": dice, "BEST DICE": best_dice})
        
        if epoch > epochs//2 and check_patience >= patience:
            break

    print(f"Best performance at epoch: {best_epoch} >> {best_dice:.4f}")


def validation(epoch, model, data_loader, criterion, classes, thr=0.5):
    print(f"Start validation #{epoch:2d}")
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)['out']

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

            wandb.log({"Val loss": loss.item()})

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [f"{c:<12}: {d.item():.4f}" for c, d in zip(classes, dices_per_class)]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    # dice_dict = {c: d.item() for c, d in zip(classes, dices_per_class)}
    # wandb.log({"Val dice": avg_dice, **dice_dict})

    return avg_dice

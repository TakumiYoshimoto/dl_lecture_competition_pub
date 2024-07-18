
# 脳磁図（MEG: Magnetoencephalography）データを用いた、分類タスクのためのPyTorchトレーニングスクリプト
# Hydraを使って設定管理を行う
# WandBを使ってトレーニングの進行状況を記録する

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import TransformerClassifier
from src.utils import set_seed


# hydra.mainデコレーターは、Hydraを用いて設定を管理するためのもの
# config_pathには、設定ファイルが格納されているディレクトリ名を指定
# config_nameには、設定ファイル名を指定
@hydra.main(version_base=None, config_path="configs", config_name="config")

def run(args: DictConfig):

    set_seed(args.seed)

    # logdirは、path（outputs/日時/時刻）を表すことになる
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    

    # wandbの初期設定（config.yamlで、use_wandbがTrueの場合のみ使用される）
    if args.use_wandb:
        wandb.init(
            mode="online", 
            dir=logdir, 
            project="MEG-classification-project", # プロジェクト名を設定
            # name=f"run",
        )


    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)

    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader   = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)

    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader  = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )



    # ------------------
    #       Model
    # ------------------
    model = TransformerClassifier(
        num_classes=train_set.num_classes,
        seq_len=train_set.seq_len,
        num_channels=train_set.num_channels,
    ).to(args.device)



    # ------------------
    #     Optimizer
    # ------------------
    # Adamオプティマイザを設定
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)



    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):

        # 出力の1行目
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        

        # 出力の2行目
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())


        # 出力の3行目
        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())


        # 出力の4行目
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        # 最新モデルの保存
        # pytorch公式ドキュメントによると、state_dict()で辞書化して保存することが推奨されている
        # logdir = "outputs/日時/時刻"
        # os.path.joinで、path名（logdir）とファイル名（model_last.pt）を結合
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        

        # wandbによる評価値の記録（config.yamlで、use_wandbがTrueの場合のみ使用される）
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        

        # val_acc配列の平均値が、それまでの最大値を上回った場合
        if np.mean(val_acc) > max_val_acc:
            # 出力の5行目
            cprint("New best.", "cyan")
            # 最良モデルの保存
            # pytorch公式ドキュメントによると、state_dict()で辞書化して保存することが推奨されている
            # logdir = "outputs/日時/時刻"
            # os.path.joinで、path名（logdir）とファイル名（model_best.pt）を結合
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            # 最大値に、val_acc配列の平均値を代入して更新
            max_val_acc = np.mean(val_acc)
            
    

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    # 最良モデルのロード
    # logdir = "outputs/日時/時刻"
    # os.path.joinで、path名（logdir）とファイル名（model_best.pt）を結合
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []

    # 出力の1行目
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()

    # 予測値の保存
    np.save(os.path.join(logdir, "submission"), preds)
    # 出力の2行目
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")




# スクリプトが直接実行された場合に、上記のrun関数を呼び出す
if __name__ == "__main__":
    run()



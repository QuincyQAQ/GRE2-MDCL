# 依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # 或 CPU 版
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install scikit-learn

# 运行（数据集可选 cora / citeseer / pubmed）
python gre2_mdcl.py --dataset cora --epochs 400 --hidden 256 --proj 128 --heads 4 \
  --tau 1.0 --alpha 0.5 --beta 1.0 --gamma 1.0 --svd-k 64 --feature-drop 0.2 --attn-drop 0.4

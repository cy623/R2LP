# ----------------cora-----------------
python RoGNN/train.py --epochs 200 --epochs_lp 20 --hidden 64 --noise_type uniform --lp_function 2 \
--lr 0.03 --dropout 0.9 --early_stopping 40 --weight_decay 1e-7 \
--alpha 77.0 --beta 1000.0 --gamma 0.8 --delta 0.9 --norm_layers 2 \
--orders 6 --orders_func_id 2 --norm_func_id 1 --dataset cora \
--pre_noise 0.2 --alpha1 0.4 --alpha2 0.3 --alpha3 0.5 --pre_select 0.5

python RoGNN/train.py --epochs 200 --epochs_lp 20 --hidden 64 --noise_type flip --lp_function 2 \
--lr 0.03 --dropout 0.9 --early_stopping 40 --weight_decay 1e-7 \
--alpha 77.0 --beta 1000.0 --gamma 0.8 --delta 0.9 --norm_layers 2 \
--orders 6 --orders_func_id 2 --norm_func_id 1 --dataset cora \
--pre_noise 0.2 --alpha1 0.4 --alpha2 0.3 --alpha3 0.5 --pre_select 0.5

-----------------chameleon------------------
python RoGNN/train.py --epochs 200 --epochs_lp 20 --hidden 64 --noise_type uniform --lp_function 2 \
--lr 0.01 --dropout 0.1 --early_stopping 40 --weight_decay 5e-5 \
--alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 --norm_layers 2 \
--orders 1 --orders_func_id 2 --norm_func_id 1 --dataset chameleon \
--pre_noise 0.2 --alpha1 0.6 --alpha2 0.0 --alpha3 0.1 --pre_select 0.2

python RoGNN/train.py --epochs 200 --epochs_lp 20 --hidden 64 --noise_type flip --lp_function 2 \
--lr 0.01 --dropout 0.1 --early_stopping 40 --weight_decay 5e-5 \
--alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 --norm_layers 2 \
--orders 1 --orders_func_id 2 --norm_func_id 1 --dataset chameleon \
--pre_noise 0.2 --alpha1 0.6 --alpha2 0.0 --alpha3 0.1 --pre_select 0.2
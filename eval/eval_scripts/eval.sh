cd /home/zwang253/workspace/scale-up-vggt/FlashVGGT/eval

echo "EVALUATION WITH 100 FRAMES"
python eval.py --config-name dense_recon num_frames=100 save_name=dense_recon_100
sleep 10

echo "EVALUATION WITH 500 FRAMES"
python eval.py --config-name dense_recon num_frames=500 save_name=dense_recon_500
sleep 10

echo "EVALUATION WITH 1000 FRAMES"
python eval.py --config-name dense_recon num_frames=1000 save_name=dense_recon_1000
sleep 10
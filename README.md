# final
optical-flow motion

1. run the opencv-trad,you can get one frame between two frames by opencv-method,(first ,you need to change th path which in opencv-trad.py one your computer )
	python opencv-trad.py

2. run the opencv-trad,you can get one frame between two frames by raft-backward warp-method
	python opencv-raft --model=models/model_1.pth
	model_1.pth is the pretrain model for raft

3.recover_video.py can make the frames to video

4.producd_data.py extract the frames from video(you can uzip the out_video and extract the frame then run opencv-trad.py to increase the frame)

5.psnr.py test the psnr and ssim 
gan(dataset) https://drive.google.com/file/d/1hpLNfE7eJfGmEmhbfovMMtlEMug1Hvb9/view?usp=sharing


In hairnet-code

�U���nweight�Bdata(�w�gpreprocess�L)����O��Jweight��data��Ƨ����A����:
python3 main.py --mode test --weight weight/000025_weight.pt
It's just a demo for learning text2video big model now.
Base on model:DFloat11/Wan2.2-T2V-A14B-2-DF11</br>
https://huggingface.co/DFloat11/Wan2.2-T2V-A14B-2-DF11
modelscope download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local_dir ./Wan-AI/Wan2.2-T2V-A14B-Diffusers 
modelscope download DFloat11/Wan2.2-T2V-A14B-2-DF11 --local_dir ./DFloat11/Wan2.2-T2V-A14B-2-DF11
modelscope download DFloat11/Wan2.2-T2V-A14B-DF11 --local_dir ./DFloat11/Wan2.2-T2V-A14B-DF11
pip install -U dfloat11[cuda12]

Need minimized VRAM 22GB,128GB RAM

